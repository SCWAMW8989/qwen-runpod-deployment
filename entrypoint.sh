#!/usr/bin/env bash
# Runtime GPU auto-detection, CUDA capability pre-flight check, quant-tier
# selection, and vision (mmproj) wiring for
# 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF on RunPod.
set -euo pipefail

REPO="0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF"
MMPROJ_FILENAME="mmproj-Qwen3.8-27B-Q8_0.gguf"
MMPROJ_URL="https://huggingface.co/${REPO}/resolve/main/${MMPROJ_FILENAME}"

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
echo "  (there is an upstream report, ggml-org/llama.cpp#18684, of" \
     "LLAMA_CACHE alone being ignored on some code paths -- after this" \
     "pod finishes its first download, confirm the .gguf actually" \
     "landed under ${CACHE_DIR} before trusting that restarts won't" \
     "re-download it.)"

# The Dockerfile asserts /llama-server exists at BUILD time and symlinks
# it onto PATH. This runtime resolution is a second, independent check,
# not the only line of defense -- if it ever fails, the build-time
# assertion should already have caught the underlying cause first.
LLAMA_SERVER_BIN="$(command -v llama-server || echo /llama-server)"
if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "FATAL: could not find an executable llama-server at" \
       "'${LLAMA_SERVER_BIN}'. This should have been caught at build" \
       "time by the Dockerfile's assertion -- rebuild the image." >&2
  exit 1
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
# without loading any model or downloading anything -- confirmed against
# real bug reports run on this exact image. Run it before spending
# minutes and bandwidth on a 15-27 GiB model pull.
#
# Limitation, stated plainly: a device being FOUND here does not
# guarantee every later CUDA call succeeds (documented real case:
# sm_110/Thor-class hardware listing cleanly, then failing at
# cublasCreate). This catches "no device at all" with certainty and
# downgrades "wrong architecture compiled in" from a silent post-download
# crash into an early warning, not a guarantee.
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
       "not appear to have initialized against it -- refusing to proceed" \
       "and silently fall back to slow CPU inference. Full probe output:" >&2
  cat "$CUDA_PROBE_LOG" >&2
  rm -f "$CUDA_PROBE_LOG"
  exit 1
fi

SUPPORTED_CCS="7.5 8.0 8.6 8.9 9.0 12.0"
# Fixed: this command substitution is a plain (non-local) assignment, so
# under `set -e` a failing pipeline here would silently kill the whole
# script before the fallback handling below ever runs -- `local var=$(cmd)`
# masks a failing substitution's exit status via `local`'s own return
# value, but a bare top-level `VAR=$(cmd)` does not. Appending `|| true`
# to the innermost pipeline prevents a "no matches found" grep from
# aborting the script.
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
             "ggml_cuda_init found the device, but later inference calls" \
             "(e.g. cublasCreate) have been known to fail on architectures" \
             "the binary wasn't compiled for, even when the device listing" \
             "itself succeeds (documented on sm_110/Thor-class hardware)." \
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
# with its CSV output even when --format=csv,noheader is requested --
# this is a well-documented real-world nvidia-smi behavior on degraded
# hardware. Filter to numeric-only lines before doing arithmetic on them,
# since a stray warning string would abort the script under `set -e`
# arithmetic evaluation.
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

# --- Multi-GPU overhead deduction ---------------------------------------
# In llama.cpp's default layer-split mode, model weights are NOT
# duplicated across GPUs -- each GPU holds a distinct subset of layers,
# so summing raw VRAM is a reasonable starting point for weight capacity.
# What summing does NOT account for is that each additional GPU brings
# its own CUDA runtime/context overhead (roughly 1-1.5 GiB), separate
# from the ~1.0-1.5 GiB already assumed for GPU #1 in the single-GPU
# tier thresholds below. `--fit on` applies its own per-device safety
# margin at actual launch time, but that protects context/layer sizing
# GIVEN a quant that basically fits -- it can't rescue a quant tier that
# was fundamentally oversized for the real multi-GPU overhead before
# --fit ever runs. Deduct a fixed 1.5 GiB per GPU beyond the first as a
# heuristic safety margin; this is a defensible estimate based on known
# per-device CUDA context cost, not a benchmarked exact figure.
if [ "$GPU_COUNT" -gt 1 ]; then
  MULTI_GPU_PENALTY_GIB=$(( (GPU_COUNT - 1) * 3 / 2 ))
  TOTAL_GIB=$(( TOTAL_GIB - MULTI_GPU_PENALTY_GIB ))
  echo "Multi-GPU setup detected (${GPU_COUNT} cards): deducted" \
       "${MULTI_GPU_PENALTY_GIB} GiB (1.5 GiB per additional GPU) for" \
       "per-device CUDA context overhead beyond the first card." \
       "Adjusted total for tier selection: ${TOTAL_GIB} GiB."
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

echo "Selected quant tier: ${QUANT}"

: "${API_KEY:?Set the API_KEY environment variable on the RunPod pod before starting.}"

# --- Vision projector: explicit download, not repo-scanning auto-detect -
# Rather than rely on "-hf"'s own mmproj-auto repo-listing heuristic
# (which works for this specific repo -- the model author's own README
# documents a verified llama-server load test -- but is still a
# convention-based guess, not a hard guarantee), download the exact
# named projector file directly and pass it via the officially documented
# "--mmproj <local_file>" flag, confirmed compatible alongside "-hf" for
# the main model in llama.cpp's own multimodal docs. If the download
# fails (transient network issue, HF outage), degrade gracefully to
# text-only for this session rather than aborting the whole pod.
MMPROJ_FLAG="--no-mmproj"
if [ "$VISION_ACTIVE" -eq 1 ]; then
  MMPROJ_LOCAL_PATH="${CACHE_DIR}/${MMPROJ_FILENAME}"
  if [ -f "$MMPROJ_LOCAL_PATH" ]; then
    echo "Vision projector already cached at ${MMPROJ_LOCAL_PATH}."
  else
    echo "Downloading vision projector to ${MMPROJ_LOCAL_PATH}..."
    if curl -fL --retry 3 --retry-delay 5 -o "$MMPROJ_LOCAL_PATH" "$MMPROJ_URL"; then
      echo "Vision projector download succeeded."
    else
      echo "WARNING: vision projector download failed after retries." \
           "Falling back to text-only for this session. Vision can be" \
           "retried on the next pod restart, or forced off with" \
           "ENABLE_VISION=0." >&2
      rm -f "$MMPROJ_LOCAL_PATH"
      VISION_ACTIVE=0
    fi
  fi
  if [ "$VISION_ACTIVE" -eq 1 ]; then
    MMPROJ_FLAG="--mmproj ${MMPROJ_LOCAL_PATH}"
  fi
fi

echo "Vision mode: $( [ "$VISION_ACTIVE" -eq 1 ] && echo "ENABLED (--mmproj ${CACHE_DIR}/${MMPROJ_FILENAME})" || echo "disabled" )"

if [ -n "${CONTEXT_SIZE:-}" ]; then
  CTX_SIZE="$CONTEXT_SIZE"
elif [ "$VISION_ACTIVE" -eq 1 ]; then
  CTX_SIZE=16384
else
  CTX_SIZE=8192
fi

echo "== Launching llama-server =="
echo "  binary     : ${LLAMA_SERVER_BIN}"
echo "  repo/quant : ${REPO}:${QUANT}"
echo "  ctx-size   : ${CTX_SIZE}"
echo "  cache dir  : ${LLAMA_CACHE}"
echo "  gpu count  : ${GPU_COUNT}"
echo "  vision     : ${VISION_ACTIVE}"

# --n-gpu-layers is intentionally left at its default ('auto') and
# --fit on (llama-server's native default) so the engine itself decides
# how many layers fit on whatever card(s) actually showed up, leaving a
# safety margin per device. On multi-GPU pods, omitting --tensor-split
# makes llama.cpp auto-balance proportionally to each card's free memory.
exec "$LLAMA_SERVER_BIN" \
  -hf "${REPO}:${QUANT}" \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size "${CTX_SIZE}" \
  --fit on \
  --flash-attn auto \
  --jinja \
  --metrics \
  ${MMPROJ_FLAG} \
  --api-key "${API_KEY}" \
  "$@"
