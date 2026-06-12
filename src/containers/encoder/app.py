"""Slemify managed encoder service.

Serves a sentence-transformers encoder over a small HTTP API that is
wire-compatible with Hugging Face Text Embeddings Inference (TEI): a POST to
/embed with {"inputs": <str | list[str]>} returns a list of embedding vectors,
one per input.

This is a Slemify-owned, multi-arch (amd64 + arm64) CPU image. It exists
because the official TEI CPU image is published for amd64 only, while Slemify
targets Graviton (arm64) CPU nodes. The model is NOT baked into the image; it
is downloaded at startup from the Hugging Face Hub based on EMBEDDING_MODEL_NAME
and cached to the writable cache dir, so one image serves any encoder.

Used by both the classifier training job (embed train inputs) and the
classifier serving pod (embed queries at inference time).
"""

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
# Cap inputs so a pathological request can't exhaust memory. Most encoders
# truncate at 512 tokens anyway; this is a coarse character-level guard.
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "8000"))

app = FastAPI()
_model: SentenceTransformer | None = None


class EmbedRequest(BaseModel):
    # Accept either a single string or a batch, matching the TEI contract.
    inputs: str | list[str]


@app.on_event("startup")
def load_model():
    global _model
    print(f"Loading embedding model: {MODEL_NAME}", flush=True)
    _model = SentenceTransformer(MODEL_NAME, device="cpu")
    # Touch the model once so the first real request doesn't pay lazy-init cost.
    _model.encode(["warmup"], normalize_embeddings=True)
    print("Embedding model ready", flush=True)


@app.get("/health")
def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    texts = [t[:MAX_INPUT_CHARS] for t in texts]
    # normalize_embeddings=True yields unit vectors (cosine-ready). Applied
    # identically at train and inference time so the head sees a stable space.
    vectors = _model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
