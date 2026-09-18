#!/usr/bin/env bash
# Verification and telemetry script for the Qwen3.8-27B RVN Heretic
# RunPod deployment.
#
# On iSH specifically: Alpine's default shell is ash, not bash, and
# bash is NOT installed by default (confirmed -- this trips up nearly
# everyone who first tries to run a bash script on Alpine/iSH). Before
# running this script on iSH:
#   apk add bash jq
#
# Usage:
#   ./verify.sh <base_url> <api_key> [ssh_target] [image_path]
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <base_url> <api_key> [ssh_target] [image_path]" >&2
  exit 1
fi

BASE_URL="$1"
API_KEY="$2"
SSH_TARGET="${3:-}"
IMAGE_PATH="${4:-}"

if ! command -v jq >/dev/null 2>&1; then
  echo "This script needs jq. On iSH: apk add jq" >&2
  exit 1
fi

echo "== Health check =="
HEALTH=$(curl -s -o /tmp/health.json -w "%{http_code}" "${BASE_URL}/health")
cat /tmp/health.json
echo
if [ "$HEALTH" != "200" ]; then
  echo "FATAL: /health returned HTTP ${HEALTH}, expected 200." >&2
  exit 1
fi
echo "Health check passed."
echo

# NOTE: /v1/models sits in llama-server's hardcoded public_endpoints
# allowlist (confirmed against the server's own source) alongside
# /health -- it returns 200 with or without a valid Authorization
# header. This call is still useful to confirm the model loaded and
# report its ID, but it does NOT validate API_KEY. The real key check
# is the dedicated negative test below.
echo "== Model listing (note: this endpoint does not require auth) =="
MODELS_STATUS=$(curl -s -o /tmp/models.json -w "%{http_code}" \
  -H "Authorization: Bearer ${API_KEY}" \
  "${BASE_URL}/v1/models")
cat /tmp/models.json
echo
if [ "$MODELS_STATUS" != "200" ]; then
  echo "FATAL: /v1/models returned HTTP ${MODELS_STATUS} -- unexpected," \
       "since this endpoint should respond regardless of auth. The" \
       "server itself may not be up yet." >&2
  exit 1
fi
echo

echo "== API key validation (negative test) =="
BAD_KEY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer this-is-deliberately-wrong" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-rvn","messages":[{"role":"user","content":"test"}],"max_tokens":1}' \
  "${BASE_URL}/v1/chat/completions")
if [ "$BAD_KEY_STATUS" = "401" ]; then
  echo "Correctly rejected a wrong key (HTTP 401). API_KEY enforcement is active."
else
  echo "WARNING: expected HTTP 401 for a deliberately wrong key, got" \
       "${BAD_KEY_STATUS} instead. Either the server isn't enforcing" \
       "--api-key, or something else is misconfigured." >&2
fi
echo

echo "== Sample text inference: latency + tokens/sec =="
PAYLOAD='{
  "model": "qwen3.8-27b-rvn",
  "messages": [
    {"role": "user", "content": "In exactly three sentences, explain what a Gated DeltaNet layer does differently from standard attention."}
  ],
  "max_tokens": 200,
  "temperature": 0.7
}'

START_NS=$(date +%s%N)
HTTP_STATUS=$(curl -s -o /tmp/completion.json -w "%{http_code}" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  "${BASE_URL}/v1/chat/completions")
END_NS=$(date +%s%N)

ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

echo "HTTP status: ${HTTP_STATUS}"
if [ "$HTTP_STATUS" != "200" ]; then
  echo "FATAL: chat completion request failed (this call DOES validate" \
       "your real API_KEY -- a 401 here means the key you passed to this" \
       "script doesn't match --api-key on the pod)." >&2
  cat /tmp/completion.json >&2
  exit 1
fi

PROMPT_TOKENS=$(jq -r '.usage.prompt_tokens // 0' /tmp/completion.json)
COMPLETION_TOKENS=$(jq -r '.usage.completion_tokens // 0' /tmp/completion.json)
TOTAL_TOKENS=$(jq -r '.usage.total_tokens // 0' /tmp/completion.json)
REPLY=$(jq -r '.choices[0].message.content // "(no content field)"' /tmp/completion.json)

ELAPSED_S=$(awk "BEGIN { printf \"%.3f\", ${ELAPSED_MS}/1000 }")
if [ "$COMPLETION_TOKENS" -gt 0 ]; then
  TOK_PER_SEC=$(awk "BEGIN { printf \"%.2f\", ${COMPLETION_TOKENS}/${ELAPSED_S} }")
else
  TOK_PER_SEC="0.00"
fi

echo "Wall time        : ${ELAPSED_S} s"
echo "Prompt tokens     : ${PROMPT_TOKENS}"
echo "Completion tokens : ${COMPLETION_TOKENS}"
echo "Total tokens      : ${TOTAL_TOKENS}"
echo "Tokens/sec (gen)  : ${TOK_PER_SEC}"
echo
echo "Model reply:"
echo "${REPLY}"
echo

if [ -n "$IMAGE_PATH" ]; then
  if [ ! -f "$IMAGE_PATH" ]; then
    echo "WARNING: image path '${IMAGE_PATH}' not found, skipping vision test." >&2
  else
    echo "== Vision test: mmproj projector check =="
    # Fixed: MIME type is now derived from the actual file extension
    # instead of being hardcoded to image/jpeg regardless of input.
    case "${IMAGE_PATH,,}" in
      *.png)  IMG_MIME="image/png" ;;
      *.webp) IMG_MIME="image/webp" ;;
      *.gif)  IMG_MIME="image/gif" ;;
      *.jpg|*.jpeg|*) IMG_MIME="image/jpeg" ;;
    esac

    B64_IMAGE=$(base64 < "$IMAGE_PATH" | tr -d '\n')
    VISION_PAYLOAD=$(jq -n --arg img "$B64_IMAGE" --arg mime "$IMG_MIME" '{
      model: "qwen3.8-27b-rvn",
      messages: [
        {
          role: "user",
          content: [
            {type: "text", text: "Describe this image in one sentence."},
            {type: "image_url", image_url: {url: ("data:" + $mime + ";base64," + $img)}}
          ]
        }
      ],
      max_tokens: 100
    }')

    VSTART_NS=$(date +%s%N)
    VISION_STATUS=$(curl -s -o /tmp/vision.json -w "%{http_code}" \
      -H "Authorization: Bearer ${API_KEY}" \
      -H "Content-Type: application/json" \
      -d "${VISION_PAYLOAD}" \
      "${BASE_URL}/v1/chat/completions")
    VEND_NS=$(date +%s%N)
    VELAPSED_S=$(awk "BEGIN { printf \"%.3f\", (${VEND_NS}-${VSTART_NS})/1000000000 }")

    echo "HTTP status: ${VISION_STATUS}  (${VELAPSED_S}s, MIME: ${IMG_MIME})"
    if [ "$VISION_STATUS" != "200" ]; then
      echo "FATAL: vision request failed. If ENABLE_VISION was explicitly" \
           "set to 0 on the pod, this is expected. Otherwise check pod" \
           "logs for a failed mmproj download." >&2
      cat /tmp/vision.json >&2
      exit 1
    fi
    jq -r '.choices[0].message.content // "(no content field)"' /tmp/vision.json
  fi
  echo
fi

echo "== Server-side telemetry (/metrics) =="
METRICS_STATUS=$(curl -s -o /tmp/metrics.txt -w "%{http_code}" \
  -H "Authorization: Bearer ${API_KEY}" \
  "${BASE_URL}/metrics")
if [ "$METRICS_STATUS" != "200" ]; then
  echo "WARNING: /metrics returned HTTP ${METRICS_STATUS}." >&2
else
  echo "Key counters since server start:"
  grep -E '^llamacpp:(prompt_tokens_total|tokens_predicted_total|prompt_tokens_seconds|predicted_tokens_seconds|requests_processing|requests_deferred) ' \
    /tmp/metrics.txt | sed 's/^/  /' || echo "  (no matching metric lines found)"
fi
echo

if [ -n "$SSH_TARGET" ]; then
  echo "== Remote VRAM allocation (via SSH) =="
  ssh -o StrictHostKeyChecking=accept-new "${SSH_TARGET}" \
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"
fi
