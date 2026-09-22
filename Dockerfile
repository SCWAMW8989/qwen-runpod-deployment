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

# --- Dynamic binary discovery (build-time), CUDA-aware ------------------
# Confirmed empirically on 2026-09-22: this image contains more than one
# file named "llama-server". A naive "find ... | head -n 1" picked a
# non-CUDA build -- its --version output had zero mention of CUDA, while
# nvidia-smi correctly saw the GPU at runtime. A file being NAMED
# llama-server does not mean it was compiled with CUDA support.
#
# Fix: enumerate ALL matching files (written to a temp file, not piped
# directly into the while loop -- piping into a while loop creates a
# subshell in POSIX sh, which would silently discard the FOUND_BIN
# variable the moment the loop ends). For each candidate, check its
# dynamic library dependencies via ldd for a CUDA runtime reference.
# This works at BUILD time with no GPU present, unlike --version's
# device-enumeration output, which needs real GPU hardware to say
# anything CUDA-related at all.
RUN set -eu; \
    CANDIDATES_FILE="$(mktemp)"; \
    find / -xdev -maxdepth 6 -type f -name 'llama-server' 2>/dev/null > "$CANDIDATES_FILE"; \
    if [ ! -s "$CANDIDATES_FILE" ]; then \
        echo "BUILD ERROR: no file named llama-server found anywhere in" >&2; \
        echo "the base image (searched depth 6 from /)." >&2; \
        exit 1; \
    fi; \
    echo "Candidates found:"; cat "$CANDIDATES_FILE"; \
    FOUND_BIN=""; \
    while IFS= read -r c; do \
        [ -z "$c" ] && continue; \
        [ ! -x "$c" ] && chmod +x "$c" 2>/dev/null || true; \
        LDD_OUT="$(ldd "$c" 2>&1 || true)"; \
        echo "-- ldd for ${c}:"; echo "$LDD_OUT"; \
        if echo "$LDD_OUT" | grep -qiE 'libcudart|libcublas|libcuda\.so'; then \
            FOUND_BIN="$c"; \
            echo "Selected CUDA-linked binary: ${FOUND_BIN}"; \
            break; \
        fi; \
    done < "$CANDIDATES_FILE"; \
    rm -f "$CANDIDATES_FILE"; \
    if [ -z "$FOUND_BIN" ]; then \
        echo "BUILD ERROR: none of the candidates are linked against a" >&2; \
        echo "CUDA runtime library (checked for libcudart, libcublas," >&2; \
        echo "libcuda.so via ldd). This base image tag may not actually" >&2; \
        echo "ship a CUDA-enabled llama-server right now." >&2; \
        exit 1; \
    fi; \
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
