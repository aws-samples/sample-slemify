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


def _detect_threads() -> int:
    """Threads to use for CPU inference, detected at startup so the pod adapts to
    whatever CPU it's actually given (no hardcoded magic number).

    Priority:
      1. RERANKER_THREADS env  - explicit operator override.
      2. cgroup CPU quota      - the container's CPU *limit*, if one is set.
      3. CPUs the process may run on (sched_getaffinity) - i.e. the node's cores
         when there is no limit.
    """
    override = os.environ.get("RERANKER_THREADS", "")
    if override.isdigit() and int(override) > 0:
        return int(override)
    # cgroup v2: "<quota> <period>" or "max <period>"
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
            if quota != "max":
                return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, \
                open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            quota, period = int(fq.read()), int(fp.read())
            if quota > 0 and period > 0:
                return max(1, round(quota / period))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


# Pin BLAS/OpenMP thread counts before torch is imported (via sentence_transformers)
# so the native libraries honor the detected value instead of spawning one thread
# per node core regardless of what this pod was scheduled with.
_THREADS = _detect_threads()
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))

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
    import torch
    torch.set_num_threads(_THREADS)
    print(f"Reranker using {_THREADS} CPU threads")
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
