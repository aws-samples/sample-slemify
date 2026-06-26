"""Classification: triage category (the Slemify-trained CPU classifier) and the
question intent.

INTENT is a question-routing signal: does the user explicitly want the agent to
act on their live cluster (inspect/validate/diagnose resources), or is this a
question to answer from the knowledge base? The default is "answer" — tools are
opt-in (explicit request, or proposed-and-confirmed / autopilot).

NOTE: classify_intent is currently a small LLM stand-in. The long-term plan is to
fold this into the triage classifier so one CPU model emits category + intent.
The graph treats intent as a pluggable input, so swapping the implementation
needs no graph change.
"""
import re

import httpx

from . import config
from . import prompts

_VALID_CATEGORIES = {
    "karpenter_config", "keda_config", "hpa_config",
    "pdb_disruption", "spot_interruption", "multi_resource", "noise",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def classify(text: str) -> dict:
    """Triage via the Slemify-trained classifier: {category, confidence}."""
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": prompts.triage_prompt(text)}],
        "max_tokens": 32,
        "temperature": 0.1,
    }
    with httpx.Client(timeout=10) as client:
        raw = client.post(f"{config.TRIAGE_URL}/v1/chat/completions",
                          json=body).json()["choices"][0]["message"]["content"]

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


_INTENT_PROMPT = """Decide whether the user is explicitly asking the assistant to act on their LIVE Kubernetes cluster (inspect, list, describe, validate, or diagnose actual resources / their current state), versus asking a question to be answered from documentation.

Reply with one word:
- "action" if they explicitly ask to check/validate/diagnose/look at their cluster or its resources.
- "answer" otherwise (concept questions, "is this config correct", "what does X do", explanations).

USER MESSAGE:
{text}

One word:"""


def classify_intent(text: str) -> str:
    """Return "action" (explicit cluster request) or "answer" (docs-first).

    Stand-in for a trained triage intent head; uses a small LLM classification.
    Defaults to "answer" on any error so the agent never reaches for tools
    unless it is confident the user asked.
    """
    try:
        resp = config.bedrock.converse(
            modelId=config.LLM_MODEL,
            messages=[{"role": "user", "content": [{"text": _INTENT_PROMPT.format(text=text[:2000])}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0},
        )
        out = resp["output"]["message"]["content"][0]["text"].lower()
        return "action" if re.search(r"\baction\b", out) else "answer"
    except Exception:
        return "answer"
