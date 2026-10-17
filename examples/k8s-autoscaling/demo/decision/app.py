"""In-cluster decision model for the K8s Autoscaling demo: the faithfulness
gate on CPU when the orchestrator runs with GATE=decision.

Serves Vela 2.0 0.3B (vLLM Semantic Router / KR Labs), a "System One" decision
model: it answers a typed question about a state in one forward pass, with a
probability, and never generates text. Here the state is the gate's input
(question, retrieved evidence, draft answer) and the question is a Span
question: which spans of the answer are not supported by the evidence?

API:
  POST /check {"question": <str>, "evidence": <str>, "answer": <str>}
  -> {"p_unsupported": <float>, "spans": [{"text", "start", "end", "probability"}],
      "input_tokens": <int>, "ms": <int>}
  GET /health -> {"status": "ok", "precision": "bf16"|"fp32", "threads": <int>}

Engine and precision: PyTorch eager (oneDNN / Arm Compute Library) with the
encoder weights in BF16 when the CPU has native BF16 matrix instructions
(Graviton3 and later, Intel Sapphire Rapids and later with AMX, AMD Zen 4 and
later), FP32 otherwise; the readout heads cast to FP32 themselves. Measured on
c9g.8xlarge (Graviton5) over 24 gate-sized checks (~2,400 tokens each), USD per
1,000 checks at the on-demand per-vCPU price:

  engine                             vCPU  p50     USD/1k  max |dp| vs FP32
  PyTorch FP32                          8  1.91 s  0.185   reference
  PyTorch BF16 encoder (this)           4  1.29 s  0.062   0.04
  PyTorch BF16 encoder                  8  0.77 s  0.075   0.04
  PyTorch BF16 + torch.compile          8  0.79 s  0.077   0.04
  ONNX Runtime FP32                     8  2.61 s  0.252   0.00
  ONNX Runtime, Arm BF16 fast-math      4  3.35 s  0.162   0.75

Four vCPU is the lowest cost per check; eight halves the latency for 20% more.
INT8 (dynamic, ONNX Runtime or PyTorch) was faster on short inputs, slower on
gate-sized ones, and moved probabilities by up to 0.7, and the gate is a
threshold on this probability, so it is not used. vLLM is not used either: it
speeds up token generation, which this model does not do, and it cannot run
Vela's span heads. Set DECISION_PRECISION=fp32|bf16 to override the detection.

The model is baked into the image at a pinned revision, so the pod needs no
internet access at runtime.
"""

import os
import sys
import time


def _detect_threads() -> int:
    """CPU threads for inference: DECISION_THREADS, else the cgroup CPU limit,
    else the CPUs this process may run on (same policy as the re-ranker)."""
    override = os.environ.get("DECISION_THREADS", "")
    if override.isdigit() and int(override) > 0:
        return int(override)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
            if quota != "max":
                return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _cpu_has_bf16() -> bool:
    """True when the CPU advertises native BF16 matrix support."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = set(f.read().split())
    except OSError:
        return False
    return bool(flags & {"bf16", "svebf16", "avx512_bf16", "amx_bf16"})


_THREADS = _detect_threads()
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))

import torch  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

MODEL_DIR = os.environ.get("DECISION_MODEL_DIR", "/app/model")
sys.path.insert(0, MODEL_DIR)
from vela2_inference import Vela2  # noqa: E402  (shipped with the pinned model)

# Field caps match what the faithfulness gate reads (agent/gate.py), so the two
# judges see the same text.
MAX_QUESTION_CHARS = 2000
MAX_EVIDENCE_CHARS = 12000
MAX_ANSWER_CHARS = 4000

QUESTION = {
    "halu": {
        "type": "span",
        "instructions": "Which spans of the answer are not supported by the context?",
        "criteria": {"unsupported": "a claim not supported by the context"},
    }
}

app = FastAPI()
_engine = None
_precision = "fp32"


class CheckRequest(BaseModel):
    question: str
    evidence: str
    answer: str


@app.on_event("startup")
def load_model():
    global _engine, _precision
    torch.set_num_threads(_THREADS)
    wanted = os.environ.get("DECISION_PRECISION", "auto").lower()
    use_bf16 = wanted == "bf16" or (wanted == "auto" and _cpu_has_bf16())
    _engine = Vela2(MODEL_DIR, backend="torch", threads=_THREADS)
    if use_bf16:
        _engine.run.model.encoder.to(torch.bfloat16)
        _precision = "bf16"
    print(
        f"Decision model ready: Vela 2.0 0.3B, {_precision}, {_THREADS} threads",
        flush=True,
    )
    with torch.inference_mode():  # first call pays kernel selection
        _engine.system_one(
            {"request": "warmup", "source": "warmup", "answer": "warmup"}, QUESTION
        )


@app.get("/health")
def health():
    if _engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "precision": _precision, "threads": _THREADS}


@app.post("/check")
def check(req: CheckRequest):
    state = {
        "request": req.question[:MAX_QUESTION_CHARS],
        "source": req.evidence[:MAX_EVIDENCE_CHARS] or "(none)",
        "answer": req.answer[:MAX_ANSWER_CHARS] or "(empty)",
    }
    t0 = time.perf_counter()
    with torch.inference_mode():
        res = _engine.system_one(state, QUESTION)
    spans = (res.get("spans") or {}).get("halu") or []
    return {
        "p_unsupported": float(res["answers"]["halu"]["noul"]),
        "spans": [
            {
                "text": s["text"],
                "start": s["start"],
                "end": s["end"],
                "probability": round(float(s["probability"]), 4),
            }
            for s in spans
        ],
        "input_tokens": res["usage"]["input_tokens"],
        "ms": round((time.perf_counter() - t0) * 1000),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
