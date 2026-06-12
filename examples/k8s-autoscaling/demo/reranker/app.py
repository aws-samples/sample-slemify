"""In-cluster cross-encoder re-ranker for the K8s Autoscaling demo.

Scores (query, document) pairs with a cross-encoder and returns relevance
scores, so the orchestrator can retrieve a wide candidate set from OpenSearch
(k=10) and keep only the best few for the auditor SLM. A cross-encoder reads
the query and document together, which ranks relevance far more accurately
than the embedding cosine similarity used for the initial vector search.

API:
  POST /rerank {"query": <str>, "documents": [<str>, ...], "top_k": <int?>}
  -> {"results": [{"index": <int>, "score": <float>}, ...]}  (sorted, best first)

The model is baked into the image at build time so the pod needs no internet
access at runtime.

Usage:
  pip install fastapi uvicorn sentence-transformers
  python3 app.py
"""

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
# Coarse guard so a pathological request can't exhaust memory before tokenizing.
MAX_DOC_CHARS = 8000
MAX_DOCS = 50

app = FastAPI()
_model: CrossEncoder | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int | None = None


@app.on_event("startup")
def load_model():
    global _model
    print(f"Loading reranker model: {MODEL_NAME}")
    _model = CrossEncoder(MODEL_NAME, device="cpu")
    # Touch the model once so the first real request doesn't pay the lazy-init cost.
    _model.predict([("warmup query", "warmup document")])
    print("Reranker model ready")


@app.get("/health")
def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post("/rerank")
def rerank(req: RerankRequest):
    docs = [d[:MAX_DOC_CHARS] for d in req.documents[:MAX_DOCS]]
    if not docs:
        return {"results": []}

    pairs = [(req.query, d) for d in docs]
    scores = _model.predict(pairs)

    ranked = sorted(
        ({"index": i, "score": float(s)} for i, s in enumerate(scores)),
        key=lambda r: r["score"],
        reverse=True,
    )
    if req.top_k is not None:
        ranked = ranked[: req.top_k]
    return {"results": ranked}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
