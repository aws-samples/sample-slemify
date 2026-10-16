"""Slemify encoder serving (ONNX, CPU, no torch).

Loads a project's encoder ONNX graph + tokenizer from S3 at startup and serves
one of three encoder-family heads, dispatched by the head.json `type`:

  - classification: returns "<label>|<confidence>" over /v1/chat/completions
    (a drop-in for the generative triage SLM).
  - regression (scoring): returns a single number in [0,1] over chat-completions
    and a native /score endpoint.
  - embedding: no head.json — returns the L2-normalized vector over a native
    /embed endpoint (TEI-compatible) and over chat-completions as JSON.

Embeds with onnxruntime + tokenizers (no torch, no sentence-transformers), so
the image is small and starts fast.

Config via environment variables:
  S3_BUCKET, PROJECT, TASK, CONF_HIGH, CONF_MEDIUM, MAX_INPUT_CHARS,
  ORT_INTRA_OP_THREADS, ORT_ALLOW_SPINNING

GET /stats returns the process CPU seconds and the request count, so a
benchmark can price the pod on the cores it actually used.
"""
import json
import math
import os
import resource
import threading

import numpy as np
import onnxruntime as ort
import extract_common as ec
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

S3_BUCKET = os.environ.get("S3_BUCKET", "")
PROJECT = os.environ.get("PROJECT", "")
TASK = os.environ.get("TASK", "")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "8000"))
CONF_HIGH = float(os.environ.get("CONF_HIGH", "0.8"))
CONF_MEDIUM = float(os.environ.get("CONF_MEDIUM", "0.5"))
# ONNX Runtime threading. The default (one thread per core the process can see,
# with spin-waiting between runs) lets a pod that requests 2 vCPU keep every
# core of a 32-vCPU node busy. Slemify sets ORT_INTRA_OP_THREADS to the pod's
# CPU request; 0 means the ONNX Runtime default. Spinning is off unless asked.
ORT_INTRA_OP_THREADS = int(os.environ.get("ORT_INTRA_OP_THREADS", "0") or 0)
ORT_ALLOW_SPINNING = os.environ.get("ORT_ALLOW_SPINNING", "0")

ARTIFACT_DIR = "/tmp/model"

app = FastAPI()
_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None
_head: dict | None = None
_input_names: set[str] = set()
_encoder_info: dict = {}
_requests = 0
_requests_lock = threading.Lock()


def _count_request():
    global _requests
    with _requests_lock:
        _requests += 1


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


def _download_artifacts():
    import boto3
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    s3 = boto3.client("s3")
    prefix = f"models/{PROJECT}"
    # Embedding models have no trained head (the encoder itself is the model);
    # classification/scoring ship a head.json. Extraction ships ONLY head.json
    # (a self-contained feature tagger) — no encoder.onnx / tokenizer.json.
    if TASK == "extraction":
        files = ["head.json"]
    elif TASK == "embedding":
        files = ["encoder.onnx", "tokenizer.json"]
    else:
        files = ["encoder.onnx", "tokenizer.json", "head.json"]
    for fname in files:
        dst = os.path.join(ARTIFACT_DIR, fname)
        s3.download_file(S3_BUCKET, f"{prefix}/{fname}", dst)
    # encoder.json (written by newer trainers) records the encoder precision.
    if TASK != "extraction":
        try:
            s3.download_file(S3_BUCKET, f"{prefix}/encoder.json",
                             os.path.join(ARTIFACT_DIR, "encoder.json"))
        except Exception:
            pass
    print(f"Downloaded artifacts from s3://{S3_BUCKET}/{prefix}/", flush=True)


@app.on_event("startup")
def load():
    global _session, _tokenizer, _head, _input_names, _encoder_info
    if not S3_BUCKET or not PROJECT:
        raise RuntimeError("S3_BUCKET and PROJECT are required")

    _download_artifacts()

    # Extraction is a self-contained feature tagger (no encoder/tokenizer).
    if TASK == "extraction":
        with open(os.path.join(ARTIFACT_DIR, "head.json")) as f:
            _head = json.load(f)
        try:
            _extract("warmup")
            print("Inference path warmed up", flush=True)
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}", flush=True)
        print(f"Extraction tagger ready: {len(_head.get('entity_types', []))} entity types",
              flush=True)
        return

    _tokenizer = Tokenizer.from_file(os.path.join(ARTIFACT_DIR, "tokenizer.json"))
    _tokenizer.enable_truncation(max_length=512)
    _tokenizer.enable_padding()

    so = ort.SessionOptions()
    if ORT_INTRA_OP_THREADS > 0:
        so.intra_op_num_threads = ORT_INTRA_OP_THREADS
    so.add_session_config_entry("session.intra_op.allow_spinning", ORT_ALLOW_SPINNING)
    _session = ort.InferenceSession(
        os.path.join(ARTIFACT_DIR, "encoder.onnx"), sess_options=so,
        providers=["CPUExecutionProvider"])
    _input_names = {i.name for i in _session.get_inputs()}
    info_path = os.path.join(ARTIFACT_DIR, "encoder.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            _encoder_info = json.load(f)
    print(f"Encoder: {_encoder_info.get('precision', 'fp32')}, "
          f"intra-op threads {ORT_INTRA_OP_THREADS or 'default'}, "
          f"spinning {'on' if ORT_ALLOW_SPINNING == '1' else 'off'}", flush=True)

    head_path = os.path.join(ARTIFACT_DIR, "head.json")
    if os.path.exists(head_path):
        with open(head_path) as f:
            _head = json.load(f)
    else:
        # Embedding: synthesize a marker head so the readiness gate and dispatch
        # logic have a consistent shape to check.
        _head = {"type": "embedding"}

    # Warm up the inference path before marking ready. The readiness probe only
    # confirms the model is *loaded*; ONNX Runtime still pays first-run graph
    # optimization and thread-pool spin-up on the first real Run(). Doing one
    # throwaway inference here moves that cost off the first user request.
    try:
        head_type = _head.get("type")
        if head_type == "embedding":
            _embed("warmup")
        elif head_type == "regression":
            _score("warmup")
        else:
            _classify("warmup")
        print("Inference path warmed up", flush=True)
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}", flush=True)

    head_type = _head.get("type")
    if head_type == "embedding":
        print("Embedding model ready", flush=True)
    elif head_type == "regression":
        print("Scoring head ready (regression)", flush=True)
    else:
        print(f"Classifier ready: {len(_head['classes'])} classes", flush=True)


@app.get("/health")
def health():
    if _head is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    # Extraction needs no ONNX session; every other head does.
    if not _is_extraction() and _session is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Process CPU seconds (user + system, all threads) and requests served."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {"cpu_seconds": round(r.ru_utime + r.ru_stime, 3), "requests": _requests,
            "precision": _encoder_info.get("precision", "fp32") if not _is_extraction() else None,
            "intra_op_threads": ORT_INTRA_OP_THREADS or None,
            "spinning": ORT_ALLOW_SPINNING == "1"}


def _embed(text: str) -> np.ndarray:
    enc = _tokenizer.encode(text[:MAX_INPUT_CHARS])
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in _input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids)
    last_hidden = _session.run(None, feed)[0]  # [1, seq, hidden]
    # bge uses CLS pooling (first token) + L2 normalization — must match training.
    cls = last_hidden[0, 0, :]
    return cls / (np.linalg.norm(cls) + 1e-12)


def _classify(text: str) -> tuple[str, float]:
    vec = _embed(text)
    classes = _head["classes"]
    coef = np.asarray(_head["coef"], dtype=np.float32)
    intercept = np.asarray(_head["intercept"], dtype=np.float32)

    if coef.shape[0] == 1:
        z = float(coef[0] @ vec) + float(intercept[0])
        p1 = 1.0 / (1.0 + math.exp(-z))
        probs = np.array([1.0 - p1, p1])
    else:
        z = coef @ vec + intercept
        z -= z.max()
        e = np.exp(z)
        probs = e / e.sum()

    best = int(np.argmax(probs))
    return classes[best], float(probs[best])


def _score(text: str) -> float:
    """Regression inference: encoder vector · weight + intercept, clamped to range."""
    vec = _embed(text)
    coef = np.asarray(_head["coef"], dtype=np.float32)
    intercept = float(_head["intercept"])
    raw = float(coef @ vec) + intercept
    lo = float(_head.get("score_min", 0.0))
    hi = float(_head.get("score_max", 1.0))
    return max(lo, min(hi, raw))


def _is_scoring() -> bool:
    return bool(_head) and _head.get("type") == "regression"


def _is_embedding() -> bool:
    return bool(_head) and _head.get("type") == "embedding"


def _is_extraction() -> bool:
    return bool(_head) and _head.get("type") == "extraction"


def _extract(text: str) -> list[dict]:
    """Run the feature-based token tagger; returns [{"type","text"}, ...]."""
    return ec.extract(text[:MAX_INPUT_CHARS], _head)


def _confidence_word(p: float) -> str:
    if p >= CONF_HIGH:
        return "high"
    if p >= CONF_MEDIUM:
        return "medium"
    return "low"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    _count_request()
    text = req.messages[-1].content if req.messages else ""
    if _is_embedding():
        vec = _embed(text).tolist()
        # Return the vector as JSON message content for clients going through the
        # chat contract; structured form is under the slemify field.
        return {
            "choices": [
                {"index": 0,
                 "message": {"role": "assistant", "content": json.dumps(vec)},
                 "finish_reason": "stop"}
            ],
            "slemify": {"vector": vec, "dim": len(vec)},
        }
    if _is_scoring():
        score = _score(text)
        # Return the bare score as message content so the orchestrator can read
        # it the same way it reads a label — a drop-in for the chat contract.
        content = f"{score:.4f}"
        return {
            "choices": [
                {"index": 0,
                 "message": {"role": "assistant", "content": content},
                 "finish_reason": "stop"}
            ],
            "slemify": {"score": round(score, 4)},
        }
    if _is_extraction():
        spans = _extract(text)
        # JSON-encode the spans as message content for the chat contract; the
        # structured list is under the slemify field.
        return {
            "choices": [
                {"index": 0,
                 "message": {"role": "assistant", "content": json.dumps(spans)},
                 "finish_reason": "stop"}
            ],
            "slemify": {"spans": spans},
        }
    label, prob = _classify(text)
    content = f"{label}|{_confidence_word(prob)}"
    return {
        "choices": [
            {"index": 0,
             "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "slemify": {"label": label, "probability": round(prob, 4)},
    }


class ScoreRequest(BaseModel):
    input: str


@app.post("/score")
def score(req: ScoreRequest):
    """Native scoring endpoint. Returns {"score": <float in [0,1]>}."""
    _count_request()
    if not _is_scoring():
        return JSONResponse(
            {"error": "this model is a classifier, not a scoring head"},
            status_code=400)
    return {"score": round(_score(req.input), 4)}


class EmbedRequest(BaseModel):
    # Accept a single string or a batch, matching the TEI /embed contract.
    inputs: str | list[str]


@app.post("/embed")
def embed(req: EmbedRequest):
    """Native embedding endpoint (TEI-compatible).

    Returns a list of L2-normalized vectors, one per input — a drop-in for the
    demo's stock bge embedding pod, now domain-tuned.
    """
    _count_request()
    if not _is_embedding():
        return JSONResponse(
            {"error": "this model is not an embedding model"},
            status_code=400)
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    return [_embed(t).tolist() for t in texts]


class ExtractRequest(BaseModel):
    input: str


@app.post("/extract")
def extract_endpoint(req: ExtractRequest):
    """Native extraction endpoint. Returns {"spans": [{"type","text"}, ...]}."""
    _count_request()
    if not _is_extraction():
        return JSONResponse(
            {"error": "this model is not an extraction tagger"},
            status_code=400)
    return {"spans": _extract(req.input)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
