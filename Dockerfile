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
# builds of this floating tag: confirmed empirically -- a build on
# 2026-09-17 failed to find the binary at /llama-server (the location
# confirmed via multiple independent Dockerhub mirror layer listings
# earlier), and a separate mirror snapshot dated just ~2 weeks earlier
# (2026-09-05) showed it landing at /app/llama-server instead, alongside
# a very recent upstream commit ("add llama in all docker images",
# #25035) that reorganized .devops/cuda.Dockerfile. Hardcoding either
# path is the same mistake twice. Instead, search the image at BUILD
# time and symlink whatever is actually found -- this makes the image
# self-adapting to future upstream reorganizations instead of breaking
# again the next time the layout shifts. If the binary genuinely can't
# be found anywhere reasonable, fail the BUILD (free, in CI) with a
# diagnostic listing, rather than failing at runtime on a rented GPU pod.
RUN set -eu; \
    FOUND_BIN="$(find / -xdev -maxdepth 6 -type f -name 'llama-server' 2>/dev/null | head -n 1)"; \
    if [ -z "$FOUND_BIN" ]; then \
        echo "BUILD ERROR: llama-server binary not found anywhere in the" >&2; \
        echo "base image (searched depth 6 from /). Upstream layout has" >&2; \
        echo "changed beyond a simple path shift. Inspect manually with:" >&2; \
        echo "  docker run --rm --entrypoint sh ghcr.io/ggml-org/llama.cpp:server-cuda -c 'find / -xdev -iname \"*llama*\" -type f 2>/dev/null'" >&2; \
        exit 1; \
    fi; \
    if [ ! -x "$FOUND_BIN" ]; then chmod +x "$FOUND_BIN"; fi; \
    echo "Found llama-server at: ${FOUND_BIN}"; \
    ln -sf "$FOUND_BIN" /usr/local/bin/llama-server; \
    echo "$FOUND_BIN" > /usr/local/share/llama-server.origin

# Persist the GGUF cache on the RunPod Network Volume, not the
# ephemeral container disk. Set every env var llama.cpp's cache
# resolution order checks to the same path as belt-and-suspenders --
# there is a closed upstream report (ggml-org/llama.cpp#18684) of
# LLAMA_CACHE alone being ignored on some code paths.
ENV LLAMA_CACHE=/workspace/llama_cache
ENV HF_HUB_CACHE=/workspace/llama_cache
ENV HUGGINGFACE_HUB_CACHE=/workspace/llama_cache
ENV HF_HOME=/workspace/llama_cache/hf_home

# NOTE: read by the Python huggingface_hub/hf_transfer package, not by
# llama-server's own self-contained C++/libcurl -hf downloader. Almost
# certainly a no-op here. Kept because harmless.
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Defaults; override per-pod in the RunPod environment variables panel.
ENV ENABLE_VISION=1
ENV VISION_BRIDGE=standard
ENV STRICT_CUDA_CHECK=0

WORKDIR /app

# --- CRLF sanitization -------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]