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
# CORRECTED APPROACH: an earlier revision checked "ldd llama-server" for
# CUDA library references and concluded no candidate was CUDA-linked.
# That test was wrong by design -- confirmed directly against this
# exact llama.cpp version's own ggml-backend-reg.cpp source: GGML loads
# its CUDA backend as a separate plugin (libggml-cuda.so) via dlopen()
# at RUNTIME, discovered through a search path (compiled-in
# GGML_BACKEND_DIR, then the executable's own directory, then the
# current working directory, or GGML_BACKEND_PATH if set) -- it is
# never a direct link-time dependency of the main binary, so ldd could
# never have shown it regardless of which binary was correct.
#
# The real question is whether libggml-cuda.so exists anywhere in this
# image at all. This step finds the llama-server binary (there was only
# one candidate in the build that surfaced the ldd false negative), and
# separately searches for the CUDA backend plugin. If found, its
# directory is recorded so entrypoint.sh can set GGML_BACKEND_PATH
# explicitly, removing any ambiguity about executable-path-relative
# discovery. If the plugin genuinely does not exist anywhere in the
# image, the build fails outright with that fact stated plainly --
# that would mean this floating tag does not currently ship CUDA
# support, and the fix is pinning to a different build tag, not more
# logic here.
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
        echo "image. Per ggml-backend-reg.cpp, the CUDA backend is loaded" >&2; \
        echo "as a dynamic plugin, not a link-time dependency -- if the" >&2; \
        echo "plugin file itself is absent, this floating :server-cuda tag" >&2; \
        echo "does not currently ship CUDA support. Pin to a different" >&2; \
        echo "build tag (see Step 0 in the deployment guide) rather than" >&2; \
        echo "modifying this script further." >&2; \
        exit 1; \
    fi; \
    echo "Found CUDA backend plugin(s):"; echo "$CUDA_PLUGIN_MATCHES"; \
    CUDA_PLUGIN_DIR="$(dirname "$(echo "$CUDA_PLUGIN_MATCHES" | head -n 1)")"; \
    echo "$CUDA_PLUGIN_DIR" > /usr/local/share/ggml-backend-path.origin; \
    echo "Recorded GGML backend plugin directory: ${CUDA_PLUGIN_DIR}"

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
