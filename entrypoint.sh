#!/usr/bin/env bash
# Runtime GPU auto-detection, CUDA capability pre-flight check, quant-tier
# selection, and vision (mmproj) wiring for
# 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF on RunPod.
set -euo pipefail

REPO="0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF"
MMPROJ_FILENAME="mmproj-Qwen3.8-27B-Q8_0.gguf"
MMPROJ_URL="https://huggingface.co/${REPO}/resolve/main/${MMPROJ_FILENAME}"
# Confirmed present directly in this repo (629,247,008 bytes, matching
# the byte-identical copy in ggml-org/Qwen3.8-27B-GGUF) -- verified
# against the live repo file listing.

CACHE_DIR="${LLAMA_CACHE:-/workspace/llama_cache}"
mkdir -p "$CACHE_DIR"
export LLAMA_CACHE="$CACHE_DIR"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_DIR}"

echo "== Cache configuration =="
echo "  LLAMA_CACHE           = ${LLAMA_CACHE}"
echo "  HF_HUB_CACHE          = ${HF_HUB_CACHE}"
echo "  HUGGINGFACE_HUB_CACHE = ${HUGGINGFACE_HUB_CACHE}"
echo "  Existing cache contents:"
find "$CACHE_DIR" -maxdepth 3 -type f -name '*.gguf' 2>/dev/null | sed 's/^/    /' || true

# The Dockerfile searches the base image for llama-server at BUILD time
# and fails the build outright if it isn't found anywhere -- necessary
# because the upstream binary location has moved across different
# builds of this floating tag. Because the Dockerfile only ever
# completes successfully when /usr/local/bin/llama-server is a valid
# symlink to wherever the real binary landed, this runtime check simply
# trusts that symlink rather than guessing a hardcoded path.
LLAMA_SERVER_BIN="$(command -v llama-server || true)"
if [ -z "$LLAMA_SERVER_BIN" ] || [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "FATAL: llama-server not found on PATH inside the running" \
       "container. The Dockerfile's build-time discovery step fails the" \
       "build outright if the binary can't be located, so a passing" \
       "build should always produce a working symlink. If you're seeing" \
       "this, the image doesn't match the Dockerfile in this repo" \
       "(stale image pull, wrong tag, or a build that predates this fix)." >&2
  exit 1
fi
if [ -f /usr/local/share/llama-server.origin ]; then
  echo "llama-server binary resolved from: $(cat /usr/local/share/llama-server.origin)"
fi

echo "== GPU detection (nvidia-smi) =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found. This container must run on a GPU-enabled" \
       "RunPod pod with the NVIDIA runtime attached. Refusing to fall back" \
       "to CPU inference silently." >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
  echo "FATAL: nvidia-smi ran but returned no GPUs." >&2
  exit 1
}

# --- CUDA backend pre-flight check --------------------------------------
# "llama-server --version" initializes the real ggml_cuda_init path and
# prints "found N CUDA devices" / "Device X: compute capability Y.Z"
# without loading any model or downloading anything.
echo "== llama-server CUDA backend pre-flight check =="
CUDA_PROBE_LOG="$(mktemp)"
if ! timeout 30 "$LLAMA_SERVER_BIN" --version >"$CUDA_PROBE_LOG" 2>&1; then
  echo "WARNING: 'llama-server --version' exited non-zero." >&2
fi

echo "Raw ggml_cuda_init output:"
grep -E "ggml_cuda_init|Device [0-9]+:|CUDA devices" "$CUDA_PROBE_LOG" \
  | sed 's/^/  /' || true

if ! grep -q "found [0-9]\+ CUDA device" "$CUDA_PROBE_LOG"; then
  echo "FATAL: llama-server's CUDA backend never reported finding a device." \
       "nvidia-smi sees a GPU but the compiled binary's CUDA backend does" \
       "not appear to have initialized against it. Full probe output:" >&2
  cat "$CUDA_PROBE_LOG" >&2
  rm -f "$CUDA_PROBE_LOG"
  exit 1
fi

SUPPORTED_CCS="7.5 8.0 8.6 8.9 9.0 12.0"
# Fixed: appended "|| true" so a pipeline with no regex matches (grep
# returning 1) doesn't silently kill the whole script under set -e --
# a bare top-level assignment (not "local") propagates a failing
# pipeline's exit status into set -e, unlike "local var=$(cmd)" which
# masks it via local's own return value.
DETECTED_CCS="$(grep -oE 'compute capability [0-9]+\.[0-9]+' "$CUDA_PROBE_LOG" \
  | grep -oE '[0-9]+\.[0-9]+' | sort -u || true)"

if [ -z "$DETECTED_CCS" ]; then
  echo "WARNING: could not parse a compute capability out of the probe" \
       "output even though a CUDA device was reported found. Skipping the" \
       "architecture cross-check; inspect the raw output above manually." >&2
else
  echo "Detected compute capabilities: $(echo "$DETECTED_CCS" | tr '\n' ' ')"
  UNSUPPORTED_FOUND=0
  while IFS= read -r cc; do
    [ -z "$cc" ] && continue
    case " $SUPPORTED_CCS " in
      *" $cc "*) : ;;
      *)
        echo "WARNING: detected compute capability ${cc} is outside this" \
             "image's documented default build list (${SUPPORTED_CCS})." \
             "If inference errors out after this point, pin a newer base" \
             "image build that explicitly covers compute capability ${cc}." >&2
        UNSUPPORTED_FOUND=1
        ;;
    esac
  done <<< "$DETECTED_CCS"

  if [ "$UNSUPPORTED_FOUND" -eq 1 ] && [ "${STRICT_CUDA_CHECK:-0}" = "1" ]; then
    echo "FATAL: STRICT_CUDA_CHECK=1 and at least one detected compute" \
         "capability is outside the documented supported list. Aborting" \
         "before spending time on the model download." >&2
    rm -f "$CUDA_PROBE_LOG"
    exit 1
  fi
fi
rm -f "$CUDA_PROBE_LOG"

# Fixed: nvidia-smi can interleave warning text (driver/XID/ECC errors)
# with its CSV output even when --format=csv,noheader is requested.
# Filter to numeric-only lines before doing arithmetic on them.
mapfile -t GPU_MEM_MIB < <(nvidia-smi --query-gpu=memory.total \
  --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$')
GPU_COUNT=${#GPU_MEM_MIB[@]}
if [ "$GPU_COUNT" -eq 0 ]; then
  echo "FATAL: nvidia-smi produced no parseable numeric VRAM readings." \
       "This can happen when the driver reports a hardware/XID error" \
       "instead of clean device metrics. Raw output:" >&2
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits >&2 || true
  exit 1
fi

TOTAL_MIB=0
for m in "${GPU_MEM_MIB[@]}"; do
  TOTAL_MIB=$((TOTAL_MIB + m))
done
TOTAL_GIB=$(( TOTAL_MIB / 1024 ))

echo "Detected ${GPU_COUNT} GPU(s), summed VRAM: ${TOTAL_GIB} GiB (${TOTAL_MIB} MiB)"

if [ "$GPU_COUNT" -gt 1 ]; then
  # Ceiling division so the deduction never rounds DOWN below the
  # intended 1.5 GiB/extra-GPU safety margin (plain integer division
  # would under-deduct at exactly GPU_COUNT=2: (2-1)*3/2 = 1, not 1.5).
  MULTI_GPU_PENALTY_GIB=$(( ((GPU_COUNT - 1) * 3 + 1) / 2 ))
  TOTAL_GIB=$(( TOTAL_GIB - MULTI_GPU_PENALTY_GIB ))
  echo "Multi-GPU setup detected (${GPU_COUNT} cards): deducted" \
       "${MULTI_GPU_PENALTY_GIB} GiB for per-device CUDA context overhead" \
       "beyond the first card. Adjusted total for tier selection: ${TOTAL_GIB} GiB."
fi

# --- Vision toggle -------------------------------------------------------
ENABLE_VISION="${ENABLE_VISION:-1}"
VISION_BRIDGE="${VISION_BRIDGE:-standard}"
VISION_RESERVE_GIB=2

EFFECTIVE_GIB="$TOTAL_GIB"
VISION_ACTIVE=0

if [ "$ENABLE_VISION" = "1" ] || [ "$ENABLE_VISION" = "true" ]; then
  CANDIDATE_GIB=$(( TOTAL_GIB - VISION_RESERVE_GIB ))
  if [ "$CANDIDATE_GIB" -ge 9 ]; then
    EFFECTIVE_GIB="$CANDIDATE_GIB"
    VISION_ACTIVE=1
    echo "Vision active (default-on): reserving ${VISION_RESERVE_GIB} GiB for" \
         "the mmproj projector and image-encoding buffers. Effective VRAM for" \
         "quant selection: ${EFFECTIVE_GIB} GiB."
  else
    echo "WARNING: vision is on by default but ${TOTAL_GIB} GiB total VRAM" \
         "cannot spare ${VISION_RESERVE_GIB} GiB for the projector on top of" \
         "even the smallest text quant. Falling back to text-only for this" \
         "pod. Set ENABLE_VISION=0 explicitly to silence this warning." >&2
    VISION_ACTIVE=0
  fi
else
  echo "Vision explicitly disabled (ENABLE_VISION=0). Running text-only."
fi

# --- Quant tier selection -----------------------------------------------
if [ -n "${QUANT_OVERRIDE:-}" ]; then
  QUANT="$QUANT_OVERRIDE"
  echo "QUANT_OVERRIDE set: forcing quant '${QUANT}' regardless of detected VRAM."
else
  if   [ "$EFFECTIVE_GIB" -ge 32 ]; then TIER="Q8_0"
  elif [ "$EFFECTIVE_GIB" -ge 26 ]; then TIER="Q6_K"
  elif [ "$EFFECTIVE_GIB" -ge 23 ]; then TIER="Q5_K_M"
  elif [ "$EFFECTIVE_GIB" -ge 20 ]; then TIER="Q4_K_M"
  elif [ "$EFFECTIVE_GIB" -ge 17 ]; then TIER="Q3_K_M"
  elif [ "$EFFECTIVE_GIB" -ge 16 ]; then TIER="Q3_K_S"
  elif [ "$EFFECTIVE_GIB" -ge 12 ]; then TIER="IQ2_XS"
  elif [ "$EFFECTIVE_GIB" -ge 11 ]; then TIER="IQ2_XXS"
  elif [ "$EFFECTIVE_GIB" -ge 9  ]; then TIER="IQ1_S"
  else
    echo "FATAL: ${EFFECTIVE_GIB} GiB effective VRAM is below the ~9 GiB floor" \
         "required even for the smallest usable quant (IQ1_S-multilingual," \
         "~6.66 GiB weights). Attach a larger GPU, or set ENABLE_VISION=0" \
         "to claw back the vision reserve." >&2
    exit 1
  fi

  QUANT="${TIER}-multilingual"

  if [ "$VISION_ACTIVE" -eq 1 ] && [ "$VISION_BRIDGE" = "high" ]; then
    case "$TIER" in
      Q3_K_M|Q4_K_M|Q5_K_M)
        QUANT="${TIER}-multilingual-vision"
        ;;
      *)
        echo "NOTE: VISION_BRIDGE=high requested but tier '${TIER}' has no" \
             "-vision twin (only Q3_K_M/Q4_K_M/Q5_K_M do). Using the plain" \
             "quant with the projector attached; this still works per the" \
             "repo's own documentation, it just skips the bridge-precision" \
             "upgrade."
        ;;
    esac
  fi
fi

# Fixed: switched from fuzzy "-hf REPO:QUANT" matching to an explicit
# filename via --hf-file, which the llama.cpp docs confirm overrides
# quant matching entirely. This repo's own README documents multiple
# files sharing the same substring per tier (e.g. Q4_K_M-multilingual
# matches the plain, -mtp, and -vision variants), so an exact filename
# is the deterministic choice.
HF_FILE="RVN-${QUANT}.gguf"

echo "Selected quant tier: ${QUANT}  (exact file: ${HF_FILE})"

: "${API_KEY:?Set the API_KEY environment variable on the RunPod pod before starting.}"

# --- Vision projector: explicit do