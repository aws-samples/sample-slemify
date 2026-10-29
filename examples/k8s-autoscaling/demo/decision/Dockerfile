# Matches the python base image pinned by the orchestrator/reranker Dockerfiles.
FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

# CPU-only torch keeps the image small (no CUDA). Versions are the ones the
# decision model was benchmarked with on Graviton5 (see app.py).
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    "torch==2.14.1" \
    "transformers==5.17.0" \
    "tokenizers==0.23.3" \
    "safetensors==0.8.0" \
    "numpy==2.5.4" \
    "huggingface_hub==1.33.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the model into the image at a pinned revision (weights and the engine
# code that ships with them), so the pod needs no internet and starts fast.
# Apache-2.0 weights and code; the tokenizer carries the Gemma Terms of Use
# (see the model's LICENSE, NOTICE and DISTRIBUTION_TERMS.md).
ARG DECISION_MODEL=vllm-sr/Vela-2.0-0.3B
ARG DECISION_REVISION=d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${DECISION_MODEL}', revision='${DECISION_REVISION}', local_dir='/app/model', \
allow_patterns=['*.json', '*.py', '*.md', 'model.safetensors', 'LICENSE*', 'NOTICE*', 'LICENSES/*'])" \
    && rm -rf /app/model/.cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DECISION_MODEL_DIR=/app/model

COPY app.py .

EXPOSE 8080
USER 1000:1000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
