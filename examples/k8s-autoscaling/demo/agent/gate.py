"""Faithfulness gate: a capable LLM decides whether the SLM draft is supported
by the retrieved evidence. This is the judgement a lexical heuristic cannot make
(it catches confidently-wrong answers that reuse the right terms). A flagged
answer is escalated to the LLM, or — for a runtime claim — triggers a cluster
check (auto in autopilot, otherwise proposed to the user).
"""
import re

from . import config
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
            inferenceConfig={"maxTokens": 200, "temperature": 0},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        verdict_m = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
        reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        verdict = verdict_m.group(1).lower() if verdict_m else "escalate"
        reason = (reason_m.group(1) if reason_m else "")[:160]
        return verdict != "pass", reason
    except Exception as e:
        return True, f"gate error: {e}"
