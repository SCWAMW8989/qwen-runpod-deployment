#!/usr/bin/env bash
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

LLAMA_SERVER_BIN="$(command -v llama-server || true)"
if [ -z "$LLAMA_SERVER_BIN" ] || [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "FATAL: llama-server not found on PATH inside the running container." >&2
  exit 1
fi
if [ -f /usr/local/share/llama-server.origin ]; then
  echo "llama-server binary resolved from: $(cat /usr/local/share/llama-server.origin)"
fi

if [ -f /usr/local/share/ggml-backend-path.origin ]; then
  export GGML_BACKEND_PATH="$(cat /usr/local/share/ggml-backend-path.origin)"
  echo "GGML_BACKEND_PATH set to: ${GGML_BACKEND_PATH}"
else
  echo "WARNING: no recorded GGML backend plugin path found. CUDA backend discovery will rely on default search paths only." >&2
fi

echo "== GPU detection (nvidia-smi) =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found." >&2
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
cat "$CUDA_PROBE_LOG" | sed 's/^/  /'

if grep -qiE 'load_backend: failed to load|error.*backend|cannot read file data' "$CUDA_PROBE_LOG"; then
  echo "FATAL: an explicit backend-load error was reported." >&2
  rm -f "$CUDA_PROBE_LOG"
  exit 1
fi

if grep -q "found [0-9]\+ CUDA device" "$CUDA_PROBE_LOG"; then
  echo "CUDA device enumeration confirmed at --version time."
else
  echo "NOTE: no explicit 'found N CUDA device' line at --version time, and" \
       "no backend-load error either. Proceeding -- the model load itself" \
       "is the real test." >&2
fi

SUPPORTED_CCS="7.5 8.0 8.6 8.9 9.0 12.0"
DETECTED_CCS="$(grep -oE 'compute capability [0-9]+\.[0-9]+' "$CUDA_PROBE_LOG" | grep -oE '[0-9]+\.[0-9]+' | sort -u || true)"

if [ -n "$DETECTED_CCS" ]; then
  echo "Detected compute capabilities: $(echo "$DETECTED_CCS" | tr '\n' ' ')"
  UNSUPPORTED_FOUND=0
  while IFS= read -r cc; do
    [ -z "$cc" ] && continue
    case " $SUPPORTED_CCS " in
      *" $cc "*) : ;;
      *)
        echo "WARNING: detected compute capability ${cc} is outside the documented build list (${SUPPORTED_CCS})." >&2
        UNSUPPORTED_FOUND=1
        ;;
    esac
  done <<< "$DETECTED_CCS"

  if [ "$UNSUPPORTED_FOUND" -eq 1 ] && [ "${STRICT_CUDA_CHECK:-0}" = "1" ]; then
    echo "FATAL: STRICT_CUDA_CHECK=1 and an unsupported compute capability was detected." >&2
    rm -f "$CUDA_PROBE_LOG"
    exit 1
  fi
fi
rm -f "$CUDA_PROBE_LOG"

mapfile -t GPU_MEM_MIB < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$')
GPU_COUNT=${#GPU_MEM_MIB[@]}
if [ "$GPU_COUNT" -eq 0 ]; then
  echo "FATAL: nvidia-smi produced no parseable numeric VRAM readings." >&2
  exit 1
fi

TOTAL_MIB=0
for m in "${GPU_MEM_MIB[@]}"; do
  TOTAL_MIB=$((TOTAL_MIB + m))
done
TOTAL_GIB=$(( TOTAL_MIB / 1024 ))

echo "Detected ${GPU_COUNT} GPU(s), summed VRAM: ${TOTAL_GIB} GiB (${TOTAL_MIB} MiB)"

if [ "$GPU_COUNT" -gt 1 ]; then
  MULTI_GPU_PENALTY_GIB=$(( ((GPU_COUNT - 1) * 3 + 1) / 2 ))
  TOTAL_GIB=$(( TOTAL_GIB - MULTI_GPU_PENALTY_GIB ))
  echo "Multi-GPU setup detected (${GPU_COUNT} cards): deducted ${MULTI_GPU_PENALTY_GIB} GiB. Adjusted total: ${TOTAL_GIB} GiB."
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
    echo "Vision active: reserving ${VISION_RESERVE_GIB} GiB. Effective VRAM: ${EFFECTIVE_GIB} GiB."
  else
    echo "WARNING: vision is on by default but VRAM cannot spare the reserve. Falling back to text-only." >&2
    VISION_ACTIVE=0
  fi
else
  echo "Vision explicitly disabled (ENABLE_VISION=0). Running text-only."
fi

if [ -n "${QUANT_OVERRIDE:-}" ]; then
  QUANT="$QUANT_OVERRIDE"
  echo "QUANT_OVERRIDE set: forcing quant '${QUANT}'."
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
        echo "NOTE: VISION_BRIDGE=high has no -vision twin for tier '${TIER}'."
        ;;
    esac
  fi
fi

HF_FILE="RVN-${QUANT}.gguf"
MODEL_URL="https://huggingface.co/${REPO}/resolve/main/${HF_FILE}"
MODEL_LOCAL_PATH="${CACHE_DIR}/${HF_FILE}"

echo "Selected quant tier: ${QUANT}  (exact file: ${HF_FILE})"

: "${API_KEY:?Set the API_KEY environment variable on the RunPod pod before starting.}"

# Fixed: pre-download the main GGUF with our own curl + retry logic
# instead of passing -hf/--hf-file and relying on llama-server's
# internal downloader. Confirmed failure mode: a transient DNS/network
# hiccup (the same kind that briefly hit the much smaller mmproj
# download and self-recovered via our retry flags) caused the internal
# downloader to hang indefinitely with zero log output and no timeout,
# for many minutes, on a multi-GB file. Using the same curl mechanism
# that already works reliably for the projector removes this entire
# class of failure for the main model too.
if [ -f "$MODEL_LOCAL_PATH" ]; then
  echo "Main model already cached at ${MODEL_LOCAL_PATH}."
else
  echo "Downloading main model to ${MODEL_LOCAL_PATH} (this is a large file, expect several minutes)..."
  if ! curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 --max-time 3600 \
       -o "$MODEL_LOCAL_PATH" "$MODEL_URL"; then
    echo "FATAL: main model download failed after retries. Check network" >&2
    echo "connectivity from this pod to huggingface.co." >&2
    rm -f "$MODEL_LOCAL_PATH"
    exit 1
  fi
  echo "Main model download succeeded."
fi

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
      echo "WARNING: vision projector download failed after retries. Falling back to text-only." >&2
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
echo "  model file : ${MODEL_LOCAL_PATH}"
echo "  ctx-size   : ${CTX_SIZE}"
echo "  cache dir  : ${LLAMA_CACHE}"
echo "  gpu count  : ${GPU_COUNT}"
echo "  vision     : ${VISION_ACTIVE}"

exec "$LLAMA_SERVER_BIN" \
  -m "${MODEL_LOCAL_PATH}" \
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
