#!/usr/bin/env bash
# Verification and telemetry script for the Qwen3.8-27B RVN Heretic
# RunPod deployment. Run from iSH, macOS, or any terminal with curl + jq.
#
# Usage:
#   ./verify.sh <base_url> <api_key> [ssh_target] [image_path]
#
#   image_path is optional. If given and the pod was started with
#   ENABLE_VISION=1 (the default), this also sends a base64-encoded
#   vision request to confirm the mmproj projector actually loaded.
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

echo "== Authenticated model listing =="
MODELS_STATUS=$(curl -s -o /tmp/models.json -w "%{http_code}" \
  -H "Authorization: Bearer ${API_KEY}" \
  "${BASE_URL}/v1/models")
cat /tmp/models.json
echo
if [ "$MODELS_STATUS" != "200" ]; then
  echo "FATAL: /v1/models returned HTTP ${MODELS_STATUS}. Check API_KEY." >&2
  exit 1
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
  echo "FATAL: chat completion request failed." >&2
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
    B64_IMAGE=$(base64 < "$IMAGE_PATH" | tr -d '\n')
    VISION_PAYLOAD=$(jq -n --arg img "$B64_IMAGE" '{
      model: "qwen3.8-27b-rvn",
      messages: [
        {
          role: "user",
          content: [
            {type: "text", text: "Describe this image in one sentence."},
            {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $img)}}
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

    echo "HTTP status: ${VISION_STATUS}  (${VELAPSED_S}s)"
    if [ "$VISION_STATUS" != "200" ]; then
      echo "FATAL: vision request failed. If ENABLE_VISION was explicitly set" \
           "to 0 on the pod, this is expected: the projector was never" \
           "loaded. Otherwise check pod logs for a failed mmproj download" \
           "(entrypoint.sh degrades to text-only on download failure, which" \
           "would also explain a failure here)." >&2
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
  echo "WARNING: /metrics returned HTTP ${METRICS_STATUS}. entrypoint.sh" \
       "passes --metrics unconditionally, so this endpoint should normally" \
       "be reachable -- a non-200 here doesn't block using the model, but" \
       "is worth investigating if you rely on this for monitoring." >&2
else
  echo "Key counters since server start:"
  grep -E '^llamacpp:(prompt_tokens_total|tokens_predicted_total|prompt_tokens_seconds|predicted_tokens_seconds|requests_processing|requests_deferred) ' \
    /tmp/metrics.txt | sed 's/^/  /' || echo "  (no matching metric lines found -- raw output saved to /tmp/metrics.txt)"
fi
echo

if [ -n "$SSH_TARGET" ]; then
  echo "== Remote VRAM allocation (via SSH) =="
  ssh -o StrictHostKeyChecking=accept-new "${SSH_TARGET}" \
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"
fi
