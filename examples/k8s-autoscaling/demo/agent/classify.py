"""Classification: the triage category and the question intent.

The TRIAGE seat (config.TRIAGE) decides who classifies:
  - "classifier": the Slemify-trained encoder + head, served as ONNX on CPU.
  - "llm": the frontier model on Bedrock, given the same prompt.
Both return the same `label|confidence` line and go through the same parser, so
the two are scored identically by the eval and differ only in who answered.

INTENT is a question-routing signal: does the user explicitly want the agent to
act on their live cluster (inspect/validate/diagnose resources), or is this a
question to answer from the knowledge base? The default is "answer": tools are
opt-in (explicit request, or proposed-and-confirmed / autopilot). With the
classifier in the triage seat, intent is a plain-code heuristic (no model call);
with the LLM in the seat, the LLM decides. Either way the graph treats intent as
a pluggable input.
"""
import re

import httpx

from . import config
from . import extract
from . import metrics
from . import prompts

_VALID_CATEGORIES = {
    "karpenter_config", "keda_config", "hpa_config",
    "pdb_disruption", "spot_interruption", "multi_resource", "noise",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def _parse(raw: str) -> dict:
    """Parse a `label|confidence` line into {category, confidence}. Tolerant of
    prose around it (the LLM sometimes adds a sentence; the classifier never
    does), and maps clear off-topic language to noise."""
    category, confidence = "unknown", "unknown"
    for part in (p.strip().lower() for p in raw.split("\n")[0].split("|") if p.strip()):
        if part in _VALID_CATEGORIES:
            category = part
        elif part in _VALID_CONFIDENCE:
            confidence = part
    if category == "unknown":
        low = raw.lower()
        for cat in _VALID_CATEGORIES:
            if cat in low:
                category = cat
                break
        if category == "unknown" and any(w in low for w in ("not relate", "unrelated", "off-topic", "noise")):
            category, confidence = "noise", "high"
    return {"category": category, "confidence": confidence}


def _classify_classifier(text: str) -> str:
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": prompts.triage_prompt(text)}],
        "max_tokens": 32,
        "temperature": 0.1,
    }
    with httpx.Client(timeout=10) as client:
        return client.post(f"{config.TRIAGE_URL}/v1/chat/completions",
                           json=body).json()["choices"][0]["message"]["content"]


def _classify_llm(text: str) -> str:
    resp = config.bedrock.converse(
        modelId=config.LLM_MODEL,
        messages=[{"role": "user", "content": [{"text": prompts.triage_prompt(text)}]}],
        inferenceConfig={"maxTokens": 32, "temperature": 0},
    )
    metrics.charge(config.LLM_MODEL, "triage", *metrics.usage_from_converse(resp))
    return resp["output"]["message"]["content"][0]["text"]


def classify(text: str) -> dict:
    """Triage via whoever holds the seat: {category, confidence}."""
    raw = _classify_llm(text) if config.TRIAGE == "llm" else _classify_classifier(text)
    return _parse(raw)


_INTENT_PROMPT = """Decide whether the user is explicitly asking the assistant to act on their LIVE Kubernetes cluster (inspect, list, describe, validate, or diagnose actual resources / their current state), versus asking a question to be answered from documentation.

Reply with one word:
- "action" if they explicitly ask to check/validate/diagnose/look at their cluster or its resources.
- "answer" otherwise (concept questions, "is this config correct", "what does X do", explanations).

USER MESSAGE:
{text}

One word:"""


def classify_intent(text: str) -> str:
    """Return "action" (explicit cluster request) or "answer" (docs-first).

    Classifier seat: extract.wants_cluster_action, plain code on CPU (an
    inspect-style verb plus a reference to their own cluster or resources). LLM
    seat: one short Bedrock classification. Both default to "answer" on any
    doubt or error, so the agent never reaches for tools unless the user asked.
    """
    if config.TRIAGE != "llm":
        return "action" if extract.wants_cluster_action(text) else "answer"
    try:
        resp = config.bedrock.converse(
            modelId=config.LLM_MODEL,
            messages=[{"role": "user", "content": [{"text": _INTENT_PROMPT.format(text=text[:2000])}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0},
        )
        metrics.charge(config.LLM_MODEL, "intent", *metrics.usage_from_converse(resp))
        out = resp["output"]["message"]["content"][0]["text"].lower()
        return "action" if re.search(r"\baction\b", out) else "answer"
    except Exception:
        return "answer"
