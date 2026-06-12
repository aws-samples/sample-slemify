"""Slemify classifier serving (ONNX, CPU, no torch).

Loads a project's encoder ONNX graph, tokenizer, and trained logistic head from
S3 at startup, then serves classification over an OpenAI-compatible
/v1/chat/completions endpoint returning "<label>|<confidence>" — a drop-in for
the generative triage SLM. Embeds with onnxruntime + tokenizers (no torch, no
sentence-transformers), so the image is small and starts fast.

Config via environment variables:
  S3_BUCKET, PROJECT, CONF_HIGH, CONF_MEDIUM, MAX_INPUT_CHARS
"""
import json
import math
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

S3_BUCKET = os.environ.get("S3_BUCKET", "")
PROJECT = os.environ.get("PROJECT", "")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "8000"))
CONF_HIGH = float(os.environ.get("CONF_HIGH", "0.8"))
CONF_MEDIUM = float(os.environ.get("CONF_MEDIUM", "0.5"))

ARTIFACT_DIR = "/tmp/model"

app = FastAPI()
_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None
_head: dict | None = None
_input_names: set[str] = set()


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
    for fname in ("encoder.onnx", "tokenizer.json", "head.json"):
        dst = os.path.join(ARTIFACT_DIR, fname)
        s3.download_file(S3_BUCKET, f"{prefix}/{fname}", dst)
    print(f"Downloaded artifacts from s3://{S3_BUCKET}/{prefix}/", flush=True)


@app.on_event("startup")
def load():
    global _session, _tokenizer, _head, _input_names
    if not S3_BUCKET or not PROJECT:
        raise RuntimeError("S3_BUCKET and PROJECT are required")

    _download_artifacts()

    _tokenizer = Tokenizer.from_file(os.path.join(ARTIFACT_DIR, "tokenizer.json"))
    _tokenizer.enable_truncation(max_length=512)
    _tokenizer.enable_padding()

    _session = ort.InferenceSession(
        os.path.join(ARTIFACT_DIR, "encoder.onnx"),
        providers=["CPUExecutionProvider"])
    _input_names = {i.name for i in _session.get_inputs()}

    with open(os.path.join(ARTIFACT_DIR, "head.json")) as f:
        _head = json.load(f)

    # Warm up the inference path before marking ready. The readiness probe only
    # confirms the model is *loaded*; ONNX Runtime still pays first-run graph
    # optimization and thread-pool spin-up on the first real Run(). Doing one
    # throwaway classification here moves that cost off the first user request.
    try:
        _classify("warmup")
        print("Inference path warmed up", flush=True)
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}", flush=True)

    print(f"Classifier ready: {len(_head['classes'])} classes", flush=True)


@app.get("/health")
def health():
    if _session is None or _head is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


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


def _confidence_word(p: float) -> str:
    if p >= CONF_HIGH:
        return "high"
    if p >= CONF_MEDIUM:
        return "medium"
    return "low"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    text = req.messages[-1].content if req.messages else ""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
