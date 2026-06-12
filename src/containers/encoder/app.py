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

Two modes (same image):
  - Encoder mode (default): serves /embed only. Used by the classifier training
    job to embed training inputs.
  - Classifier mode (when S3_BUCKET + PROJECT are set): additionally loads a
    trained logistic head (head.json) from S3 and serves an OpenAI-compatible
    /v1/chat/completions endpoint that returns "<label>|<confidence>", a drop-in
    for the generative triage SLM. Embeds in-process (no extra network hop).
"""

import json
import math
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
# Cap inputs so a pathological request can't exhaust memory. Most encoders
# truncate at 512 tokens anyway; this is a coarse character-level guard.
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "8000"))

# Classifier mode is enabled when both are set; the head is loaded from
# s3://<S3_BUCKET>/models/<PROJECT>/head.json at startup.
S3_BUCKET = os.environ.get("S3_BUCKET", "")
PROJECT = os.environ.get("PROJECT", "")
CLASSIFIER_MODE = bool(S3_BUCKET and PROJECT)

# Softmax-probability thresholds mapped to the high/medium/low confidence words
# the orchestrator expects. Tunable via env.
CONF_HIGH = float(os.environ.get("CONF_HIGH", "0.8"))
CONF_MEDIUM = float(os.environ.get("CONF_MEDIUM", "0.5"))

app = FastAPI()
_model: SentenceTransformer | None = None
_head: dict | None = None  # {classes, coef, intercept} when in classifier mode


class EmbedRequest(BaseModel):
    # Accept either a single string or a batch, matching the TEI contract.
    inputs: str | list[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # OpenAI chat-completions shape; only messages are used.
    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@app.on_event("startup")
def load_model():
    global _model, _head
    print(f"Loading embedding model: {MODEL_NAME}", flush=True)
    _model = SentenceTransformer(MODEL_NAME, device="cpu")
    # Touch the model once so the first real request doesn't pay lazy-init cost.
    _model.encode(["warmup"], normalize_embeddings=True)
    print("Embedding model ready", flush=True)

    if CLASSIFIER_MODE:
        import boto3
        key = f"models/{PROJECT}/head.json"
        print(f"Classifier mode: loading head from s3://{S3_BUCKET}/{key}", flush=True)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        _head = json.loads(obj["Body"].read())
        print(f"Head loaded: {len(_head['classes'])} classes", flush=True)


@app.get("/health")
def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    if CLASSIFIER_MODE and _head is None:
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


def _classify(text: str) -> tuple[str, float]:
    """Embed text, apply the logistic head, return (label, probability)."""
    vec = _model.encode([text[:MAX_INPUT_CHARS]], normalize_embeddings=True)[0]
    classes = _head["classes"]
    coef = _head["coef"]          # shape: [n_classes, dim] (or [1, dim] binary)
    intercept = _head["intercept"]

    # Compute decision scores per class, then softmax for a probability.
    if len(coef) == 1:
        # Binary logistic: single decision boundary -> sigmoid.
        z = sum(c * v for c, v in zip(coef[0], vec)) + intercept[0]
        p1 = 1.0 / (1.0 + math.exp(-z))
        probs = [1.0 - p1, p1]
    else:
        scores = []
        for ci in range(len(classes)):
            z = sum(c * v for c, v in zip(coef[ci], vec)) + intercept[ci]
            scores.append(z)
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]

    best = max(range(len(classes)), key=lambda i: probs[i])
    return classes[best], probs[best]


def _confidence_word(p: float) -> str:
    if p >= CONF_HIGH:
        return "high"
    if p >= CONF_MEDIUM:
        return "medium"
    return "low"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible classification endpoint (classifier mode only).

    Returns "<label>|<confidence>" as the message content, a drop-in for the
    generative triage SLM the orchestrator previously called.
    """
    if not CLASSIFIER_MODE or _head is None:
        return JSONResponse(
            {"error": "classifier mode not enabled on this encoder"},
            status_code=400,
        )

    # Use the last user message as the query (matches how training embedded
    # instruction + input concatenated; the orchestrator sends that combined).
    text = req.messages[-1].content if req.messages else ""
    label, prob = _classify(text)
    content = f"{label}|{_confidence_word(prob)}"

    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        # Surface the raw probability for clients that want the numeric score.
        "slemify": {"label": label, "probability": round(prob, 4)},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
