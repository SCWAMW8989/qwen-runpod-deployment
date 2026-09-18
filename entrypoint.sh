#!/usr/bin/env bash
# Runtime GPU auto-detection, CUDA capability pre-flight check, quant-tier
# selection, and vision (mmproj) wiring for
# 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF on RunPod.
set -euo pipefail

REPO="0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF"
MMPROJ_FILENAME="mmproj-Qwen3.8-27B-Q8_0.gguf"
MMPROJ_URL="https://huggingface.co/${REPO}/resolve/main/${MMPROJ_FILENAME}"
# Confirmed present directly in this repo (629,247,008 bytes, matching
# the byte-identical copy in ggml-org/Qwen3.8-27B-GGUF) -- 0bserverx
# mirrors the official projector into their own repo for convenience,
# and this URL has been independently re-verified against the live
# repo file listing.

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

LLAMA_SERVER_BIN="$(command -v llama-server || true)"
if [ -z "$LLAMA_SERVER_BIN" ] || [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "FATAL: llama-server not found on PATH inside the running" \
       "container. The Dockerfile's build-time discovery step fails the" \
       "build outright if the binary can't be located, so a passing" \
       "build should always produce a working symlink. If you're seeing" \
       "this, the image doesn't match the Dockerfile in this repo." >&2
  exit 1
fi
if [ -f /usr/local/share/llama-server.origin ]; then
  echo "llama-server binary resolved from: $(cat /usr/local/share/llama-server.origin)"
fi

echo "== GPU detection (nvidia-smi) =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found. This container must run on a GPU-enabled" \
       "RunPod pod with the NVIDIA runtime attached." >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
  echo "FATAL: nvidia-smi ran but returned no GPUs." >&2
  exit 1
}

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
       "Full probe output:" >&2
  cat "$CUDA_PROBE_LOG" >&2
  rm -f "$CUDA_PROBE_LOG"
  exit 1
fi

SUPPORTED_CCS="7.5 8.0 8.6 8.9 9.0 12.0"
DETECTED_CCS="$(grep -oE 'compute capability [0-9]+\.[0-9]+' "$CUDA_PROBE_LOG" \
  | grep -oE '[0-9]+\.[0-9]+' | sort -u || true)"

if [ -z "$DETECTED_CCS" ]; then
  echo "WARNING: could not parse a compute capability out of the probe output." >&2
else
  echo "Detected compute capabilities: $(echo "$DETECTED_CCS" | tr '\n' ' ')"
  UNSUPPORTED_FOUND=0
  while IFS= read -r cc; do
    [ -z "$cc" ] && continue
    case " $SUPPORTED_CCS " in
      *" $cc "*) : ;;
      *)
        echo "WARNING: detected compute capability ${cc} is outside this" \
             "image's documented default build list (${SUPPORTED_CCS})." >&2
        UNSUPPORTED_FOUND=1
        ;;
    esac
  done <<< "$DETECTED_CCS"

  if [ "$UNSUPPORTED_FOUND" -eq 1 ] && [ "${STRICT_CUDA_CHECK:-0}" = "1" ]; then
    echo "FATAL: STRICT_CUDA_CHECK=1 and at least one detected compute" \
         "capability is outside the documented supported list." >&2
    rm -f "$CUDA_PROBE_LOG"
    exit 1
  fi
fi
rm -f "$CUDA_PROBE_LOG"

mapfile -t GPU_MEM_MIB < <(nvidia-smi --query-gpu=memory.total \
  --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$')
GPU_COUNT=${#GPU_MEM_MIB[@]}
if [ "$GPU_COUNT" -eq 0 ]; then
  echo "FATAL: nvidia-smi produced no parseable numeric VRAM readings." >&2
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
  # intended 1.5 GiB/extra-GPU safety margin -- plain integer division
  # (e.g. (2-1)*3/2 = 1, not 1.5) would under-deduct at GPU_COUNT=2.
  # Rounding up here means occasionally reserving slightly more than
  # strictly necessary, which is the safe direction for a VRAM margin.
  MULTI_GPU_PENALTY_GIB=$(( ((GPU_COUNT - 1) * 3 + 1) / 2 ))
  TOTAL_GIB=$(( TOTAL_GIB - MULTI_GPU_PENALTY_GIB ))
  echo "Multi-GPU setup detected (${GPU_COUNT} cards): deducted" \
       "${MULTI_GPU_PENALTY_GIB} GiB for per-device CUDA context overhead" \
       "beyond the first card. Adjusted total: ${TOTAL_GIB} GiB."
fi

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
    echo "Vision active (default-on): reserving ${VISION_RESERVE_GIB} GiB." \
         "Effective VRAM for quant selection: ${EFFECTIVE_GIB} GiB."
  else
    echo "WARNING: vision is on by default but ${TOTAL_GIB} GiB total VRAM" \
         "cannot spare ${VISION_RESERVE_GIB} GiB. Falling back to text-only." >&2
    VISION_ACTIVE=0
  fi
else
  echo "Vision explicitly disabled (ENABLE_VISION=0). Running text-only."
fi

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
    echo "FATAL: ${EFFECTIVE_GIB} GiB effective VRAM is below the ~9 GiB floor." >&2
    exit 1
  fi

  QUANT="${TIER}-multilingual"

  if [ "$VISION_ACTIVE" -eq 1 ] && [ "$VISION_BRIDGE" = "high" ]; then
    case "$TIER" in
      Q3_K_M|Q4_K_M|Q5_K_M)
        QUANT="${TIER}-multilingual-vision"
        ;;
      *)
        echo "NOTE: VISION_BRIDGE=high has no -vision twin for tier '${TIER}'" \
             "(only Q3_K_M/Q4_K_M/Q5_K_M do). Using the plain quant."
        ;;
    esac
  fi
fi

# Fixed: switched from "-hf REPO:QUANT" fuzzy quant-matching to an
# explicit filename via --hf-file. This repo's own README documents
# multiple files sharing the same substring per tier (e.g.
# RVN-Q4_K_M-multilingual.gguf, RVN-Q4_K_M-multilingual-mtp.gguf, and
# RVN-Q4_K_M-multilingual-vision.gguf all contain "Q4_K_M-multilingual")
# and llama.cpp's own docs only describe "-hf" quant matching as
# case-insensitive without specifying a disambiguation rule when
# multiple files match. --hf-file is documented to override quant
# matching entirely, removing the ambiguity rather than trusting an
# undocumented tie-break.
HF_FILE="RVN-${QUANT}.gguf"

echo "Selected quant tier: ${QUANT}  (exact file: ${HF_FILE})"

: "${API_KEY:?Set the API_KEY environment variable on the RunPod pod before starting.}"

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
           "Falling back to text-only for this session." >&2
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
echo "  repo       : ${REPO}"
echo "  hf-file    : ${HF_FILE}"
echo "  ctx-size   : ${CTX_SIZE}"
echo "  cache dir  : ${LLAMA_CACHE}"
echo "  gpu count  : ${GPU_COUNT}"
echo "  vision     : ${VISION_ACTIVE}"

exec "$LLAMA_SERVER_BIN" \
  -hf "${REPO}" \
  --hf-file "${HF_FILE}" \
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
