# Matches the python base image pinned by the orchestrator Dockerfile.
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

# CPU-only torch keeps the image small (no CUDA). sentence-transformers pulls in
# transformers + tokenizers.
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    "sentence-transformers==3.4.1" \
    "torch==2.6.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu

ENV EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5

# Bake the model into the image so the pod needs no internet at runtime and
# starts fast. HF cache lives under /app so it ships in the image layer.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL_NAME}', device='cpu')"

# Run fully offline at runtime: model is already present, so no network or
# cache writes are needed (compatible with readOnlyRootFilesystem).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app.py .

EXPOSE 8080

USER 1000:1000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
