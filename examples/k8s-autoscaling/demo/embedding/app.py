"""In-cluster embedding service for the K8s Autoscaling demo.

Serves bge-base-en-v1.5 (768 dimensions) over a small HTTP API that is
wire-compatible with Hugging Face Text Embeddings Inference (TEI): a POST to
/embed with {"inputs": <str | list[str]>} returns a list of embedding vectors,
one per input.

This exists because the official TEI CPU image is published for amd64 only,
while this demo runs on Graviton (arm64) nodes. The model is baked into the
image at build time so the pod needs no internet access at runtime.

Usage:
  pip install fastapi uvicorn sentence-transformers
  python3 app.py
"""

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
# Cap inputs so a pathological request can't exhaust memory. bge-base truncates
# at 512 tokens anyway; this is a coarse character-level guard before tokenizing.
MAX_INPUT_CHARS = 8000

app = FastAPI()
_model: SentenceTransformer | None = None


class EmbedRequest(BaseModel):
    # Accept either a single string or a batch, matching the TEI contract.
    inputs: str | list[str]


@app.on_event("startup")
def load_model():
    global _model
    print(f"Loading embedding model: {MODEL_NAME}")
    _model = SentenceTransformer(MODEL_NAME, device="cpu")
    # Touch the model once so the first real request doesn't pay the lazy-init cost.
    _model.encode(["warmup"], normalize_embeddings=True)
    print("Embedding model ready")


@app.get("/health")
def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    texts = [t[:MAX_INPUT_CHARS] for t in texts]
    # normalize_embeddings=True yields unit vectors, matching cosine similarity
    # search in OpenSearch. Applied identically at index and query time.
    vectors = _model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
