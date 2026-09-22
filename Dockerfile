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

# --- Binary discovery + CUDA backend plugin verification (build-time) --
# Confirmed against the actual ggml-backend-reg.cpp source: GGML_BACKEND_PATH
# is passed directly into ggml_backend_load(path), which calls
# dl_load_library(path) on that EXACT string as a single file -- it is
# NOT a directory to search. An earlier revision stored the plugin's
# containing directory (via dirname) and passed that, which produced
# "failed to load /app: /app: cannot read file data: Is a directory" at
# runtime. This revision stores the full file path to libggml-cuda.so
# itself.
RUN set -eu; \
    BIN_CANDIDATES="$(find / -xdev -maxdepth 6 -type f -name 'llama-server' 2>/dev/null)"; \
    if [ -z "$BIN_CANDIDATES" ]; then \
        echo "BUILD ERROR: no file named llama-server found anywhere in the base image." >&2; \
        exit 1; \
    fi; \
    echo "llama-server candidate(s):"; echo "$BIN_CANDIDATES"; \
    FOUND_BIN="$(echo "$BIN_CANDIDATES" | head -n 1)"; \
    [ ! -x "$FOUND_BIN" ] && chmod +x "$FOUND_BIN" || true; \
    ln -sf "$FOUND_BIN" /usr/local/bin/llama-server; \
    echo "$FOUND_BIN" > /usr/local/share/llama-server.origin; \
    echo "Symlinked llama-server -> ${FOUND_BIN}"; \
    \
    CUDA_PLUGIN_MATCHES="$(find / -xdev -maxdepth 6 -type f -name 'libggml-cuda.so*' 2>/dev/null)"; \
    if [ -z "$CUDA_PLUGIN_MATCHES" ]; then \
        echo "BUILD ERROR: no libggml-cuda.so plugin found anywhere in this" >&2; \
        echo "image. This floating :server-cuda tag does not currently ship" >&2; \
        echo "CUDA support. Pin to a different build tag instead." >&2; \
        exit 1; \
    fi; \
    echo "Found CUDA backend plugin(s):"; echo "$CUDA_PLUGIN_MATCHES"; \
    CUDA_PLUGIN_FILE="$(echo "$CUDA_PLUGIN_MATCHES" | head -n 1)"; \
    echo "$CUDA_PLUGIN_FILE" > /usr/local/share/ggml-backend-path.origin; \
    echo "Recorded GGML backend plugin FILE: ${CUDA_PLUGIN_FILE}"

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
