# RunPod deployment: Qwen3.8-27B RVN Heretic Abliterated (GGUF), auto GPU
# detect, vision (mmproj) support ON by default, CUDA pre-flight check.
#
# --- PIN THIS BEFORE PRODUCTION USE ---
# ":server-cuda" is a floating tag. Run:
#   docker run --rm --gpus all ghcr.io/ggml-org/llama.cpp:server-cuda --version
# note the printed build tag (e.g. b10955), then change FROM below to the
# matching immutable tag once you've confirmed a pod boots cleanly.
# Do NOT hardcode an old/arbitrary build number without verifying it --
# builds before roughly the six-to-seven-thousand range predate the
# --fit flag this entrypoint relies on unconditionally, and older builds
# also predate this model's hybrid-architecture support entirely.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

LABEL maintainer="Stephen Whitehurst" \
      description="Qwen3.8-27B RVN Heretic Abliterated Uncensored GGUF, auto GPU-tiered llama-server on RunPod, vision enabled by default"

# --- Build-time binary path assertion --------------------------------
# The upstream image copies the binary to /llama-server (filesystem
# root) and sets ENTRYPOINT ["/llama-server"] by absolute path, confirmed
# against three independent Dockerhub mirror layer listings. Rather than
# discover a path mismatch at runtime on a rented GPU pod, fail the
# BUILD itself (free, in CI) if that assumption ever stops holding.
RUN test -f /llama-server || \
    (echo "BUILD ERROR: /llama-server not found in base image -- upstream" \
          "install location may have changed. Update this Dockerfile's" \
          "assumption before proceeding." >&2 && exit 1)
RUN ln -sf /llama-server /usr/local/bin/llama-server

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
# If entrypoint.sh is ever edited or checked out with CRLF line endings
# (Windows editors, some git autocrlf configurations), Linux's shebang
# resolution breaks with "/bin/bash^M: bad interpreter". Strip \r before
# marking it executable so this can't happen regardless of how the file
# was edited upstream of this build.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]