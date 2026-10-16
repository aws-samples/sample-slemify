"""Faithfulness gate: decides whether a draft is supported by the retrieved
evidence. This is the judgement a lexical heuristic cannot make (it catches
confidently-wrong answers that reuse the right terms). A flagged answer is
escalated to the LLM, or — for a runtime claim — triggers a cluster check (auto
in autopilot, otherwise proposed to the user).

Two judges fill the step (config.GATE):
  llm       the frontier model reads the evidence and the draft (GATE_PROMPT)
  decision  a CPU decision model (decision/app.py) scores which spans of the
            draft the evidence does not support, in one forward pass
"""
import re

import httpx

from . import config, metrics
from .prompts import GATE_PROMPT


def llm_gate(query: str, draft: str, context: str) -> tuple[bool, str]:
    """Returns (escalate, reason). On an empty draft or any gate error, escalate
    (fail safe toward the stronger model rather than ship an unchecked answer)."""
    if not draft.strip():
        return True, "empty draft"
    prompt = GATE_PROMPT.format(context=(context or "(none)")[:12000],
                                query=query[:2000], answer=draft[:4000])
    try:
        resp = config.bedrock.converse(
            modelId=config.GATE_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            # Claude Sonnet 5 rejects temperature/topP in inferenceConfig.
            inferenceConfig={"maxTokens": 200},
            additionalModelRequestFields=config.BEDROCK_EXTRA_FIELDS,
        )
        metrics.charge(config.GATE_MODEL, "gate", *metrics.usage_from_converse(resp))
        text = metrics.text_from_converse(resp)
        verdict_m = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
        reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        verdict = verdict_m.group(1).lower() if verdict_m else "escalate"
        reason = (reason_m.group(1) if reason_m else "")[:160]
        return verdict != "pass", reason
    except Exception as e:
        return True, f"gate error: {e}"


def decision_gate(query: str, draft: str, context: str) -> tuple[bool, str]:
    """Returns (escalate, reason) from the CPU decision model. The model returns
    P(the draft contains a claim the evidence does not support) and the spans it
    suspects; config.GATE_THRESHOLD turns the probability into the verdict. On an
    empty draft or any error, escalate (fail safe toward the stronger model)."""
    if not draft.strip():
        return True, "empty draft"
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{config.DECISION_URL}/check",
                               json={"question": query[:2000], "evidence": (context or "")[:12000],
                                     "answer": draft[:4000]})
            resp.raise_for_status()
            d = resp.json()
    except Exception as e:
        return True, f"gate error: {e}"
    p = float(d["p_unsupported"])
    top = max(d.get("spans") or [], key=lambda s: s["probability"], default=None)
    reason = f"P(unsupported) {p:.2f}"
    if top:
        reason += f", e.g. \u201c{top['text'][:80]}\u201d"
    return p >= config.GATE_THRESHOLD, reason


def check(query: str, draft: str, context: str) -> tuple[bool, str]:
    """The gate step: dispatch to whichever judge config.GATE names."""
    if config.GATE == "decision":
        return decision_gate(query, draft, context)
    return llm_gate(query, draft, context)
