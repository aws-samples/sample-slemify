# DEPRECATED: Training now uses the official unsloth/unsloth Docker image directly.
FROM winglian/axolotl:0.7.1

# Install llama.cpp for GGUF conversion and quantization
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /workspace/llama.cpp && \
    cd /workspace/llama.cpp && \
    cmake -B build && \
    cmake --build build --target llama-quantize -j$(nproc) && \
    cp build/bin/llama-quantize /workspace/llama.cpp/llama-quantize

# Install AWS CLI for S3 checkpoint sync
RUN pip install --no-cache-dir awscli boto3

# Copy evaluation and checkpoint scripts
COPY scripts/ /workspace/scripts/

WORKDIR /workspace

USER 1000

HEALTHCHECK NONE
