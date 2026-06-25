# Builder stage: clone llama.cpp at a pinned tag and build the quantize binary.
# build-essential + cmake are only needed here; they never reach the runtime image.
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3 AS builder

# Pin llama.cpp to a specific release tag for reproducible conversions. Must be
# recent enough to support the base architectures we convert (e.g. Qwen3).
ARG LLAMA_CPP_TAG=b9789

RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp.git llama.cpp

WORKDIR /opt/llama.cpp
# CPU-only build of just the quantize tool. Build it static (no shared libs) and
# without OpenMP so the single binary is self-contained and needs no llama.cpp
# .so files or libgomp at runtime in the slim image. Conversion (HF -> f16 GGUF)
# is done by the bundled Python script; quantize is the only compiled binary.
RUN cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF \
    && cmake --build build --target llama-quantize --config Release -j "$(nproc)"

# Runtime stage: CPU image with the conversion toolchain.
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

COPY requirements.txt .
# Install CPU torch from the PyTorch CPU index first, then the rest. Keeping the
# torch version in requirements.txt (single source of truth) prevents the second
# resolve from pulling the large default CUDA wheel.
RUN TORCH_VERSION="$(sed -n 's/^torch==//p' requirements.txt)" \
    && pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bring in the compiled quantize binary and llama.cpp's HF -> GGUF converter
# (plus the gguf-py package and the conversion/ package the converter imports,
# e.g. conversion.qwen) from the builder stage.
COPY --from=builder /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=builder /opt/llama.cpp/convert_hf_to_gguf.py /app/convert_hf_to_gguf.py
COPY --from=builder /opt/llama.cpp/conversion /app/conversion
COPY --from=builder /opt/llama.cpp/gguf-py /app/gguf-py

# Install the gguf python package the converter imports.
RUN pip install --no-cache-dir /app/gguf-py

COPY convert.py /app/convert.py

# Caches stay under the writable /tmp. USER/HOME are set so libraries that
# resolve the current user (e.g. huggingface_hub) don't fail on the numeric uid
# 1000, which has no /etc/passwd entry.
ENV HF_HOME=/tmp/hf-cache \
    HOME=/tmp \
    USER=slemify \
    PYTHONUNBUFFERED=1

USER 1000:1000

ENTRYPOINT ["python", "convert.py"]
