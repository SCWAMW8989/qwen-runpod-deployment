# RunPod deployment: Qwen3.8-27B RVN Heretic Abliterated (GGUF), auto GPU
# detect, vision (mmproj) support ON by default, CUDA pre-flight check.
#
# --- PIN THIS BEFORE PRODUCTION USE ---
# ":server-cuda" is a floating tag. Run:
#   docker run --rm --gpus all ghcr.io/ggml-org/llama.cpp:server-cuda --version
# note the printed build tag, then change FROM below to the matching
# immutable tag once you've confirmed a pod boots cleanly.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

LABEL maintainer="Stephen Whitehurst" \
      description="Qwen3.8-27B RVN Heretic Abliterated Uncensored GGUF, auto GPU-tiered llama-server on RunPod, vision enabled by default"

# --- Dynamic binary discovery (build-time) -----------------------------
# The upstream image's binary install location is NOT stable across
# builds of this floating tag: confirmed empirically, a build failed to
# find the binary at /llama-server (a location previously confirmed via
# multiple independent Dockerhub mirror layer listings), while a
# separate mirror snapshot from roughly two weeks earlier showed it at
# /app/llama-server instead. The exact upstream commit responsible for
# the most recent shift was not independently confirmed and is
# intentionally not cited here as fact. What matters functionally is
# that hardcoding either path is the same mistake twice -- instead,
# search the image at BUILD time and symlink whatever is actually
# found, so the image self-adapts to future upstream reorganizations.
RUN set -eu; \
    FOUND_BIN="$(find / -xdev -maxdepth 6 -type f -name 'llama-server' 2>/dev/null | head -n 1)"; \
    if [ -z "$FOUND_BIN" ]; then \
        echo "BUILD ERROR: llama-server binary not found anywhere in the" >&2; \
        echo "base image (searched depth 6 from /). Inspect manually with:" >&2; \
        echo "  docker run --rm --entrypoint sh ghcr.io/ggml-org/llama.cpp:server-cuda -c 'find / -xdev -iname \"*llama*\" -type f 2>/dev/null'" >&2; \
        exit 1; \
    fi; \
    if [ ! -x "$FOUND_BIN" ]; then chmod +x "$FOUND_BIN"; fi; \
    echo "Found llama-server at: ${FOUND_BIN}"; \
    ln -sf "$FOUND_BIN" /usr/local/bin/llama-server; \
    echo "$FOUND_BIN" > /usr/local/share/llama-server.origin

ENV LLAMA_CACHE=/workspace/llama_cache
ENV HF_HUB_CACHE=/workspace/llama_cache
ENV HUGGINGFACE_HUB_CACHE=/workspace/llama_cache
ENV HF_HOME=/workspace/llama_cache/hf_home
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ENV ENABLE_VISION=1
ENV VISION_BRIDGE=standard
ENV STRICT_CUDA_CHECK=0

WORKDIR /app

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
