"""All prompt text for the agent, in one place.

Every instruction string and prompt template the agent sends to a model lives
here, with small builder functions that assemble the final prompt (instruction +
framing + context + query). Centralized so the prompts can be read, reviewed, and
edited together rather than scattered across modules. Behavior is identical to the
previous inline strings; this module only collects them.

Which model sees which prompt:
  - triage_prompt   -> triage classifier SLM (ONNX) : classify intent
  - auditor_prompt  -> auditor SLM (llama.cpp, CPU)  : the grounded draft answer
  - llm_prompt      -> escalation LLM (Bedrock)      : when the gate escalates
  - calibration_prompt -> calibration LLM (Bedrock)  : the abstain backstop
  - GATE_PROMPT     -> faithfulness gate LLM (Bedrock): is the draft supported?
"""

# --- Instruction strings ---

TRIAGE_INSTRUCTION = (
    "Classify this Kubernetes autoscaling support query into a routing "
    "category and confidence level."
)

AUDITOR_INSTRUCTION = (
    "You are a Kubernetes autoscaling auditor. "
    "Answer ONLY based on the reference documentation below. "
    "Do NOT invent fields, behaviors, or modes not in the docs. "
    "If the docs don't cover something, say so. "
    "State what is correct, why, and provide a fix if needed."
)

LLM_INSTRUCTION = (
    "You are a Kubernetes autoscaling expert. Answer the user's question accurately "
    "using the provided documentation context. Be specific and include YAML examples "
    "when relevant. If the documentation doesn't cover the topic, say so."
)

CALIBRATION_INSTRUCTION = (
    "You are a Kubernetes autoscaling expert. A previous draft answer was flagged as not fully "
    "supported by the evidence. Produce a CALIBRATED answer that:\n"
    "- states ONLY what the documentation and cluster evidence below actually support;\n"
    "- explicitly says what you could NOT confirm from the evidence;\n"
    "- does NOT assert the flagged unsupported claim, and does NOT recommend changing a "
    "configuration the evidence shows is valid;\n"
    "- if the user's question assumes a problem the evidence does not show, say plainly that you "
    "could not confirm that problem.\n"
    "Being honest about uncertainty is the goal; never state something you cannot support."
)

# Template for the faithfulness gate. Formatted with context/query/answer by
# gate.llm_gate (which also truncates each field). Kept as a template here so the
# gate's wording lives alongside the other prompts.
GATE_PROMPT = """You are a faithfulness checker for a Kubernetes autoscaling assistant. Given the REFERENCE evidence, the USER QUESTION, and a proposed ANSWER, decide whether to PASS the answer or ESCALATE it to a stronger model.

ESCALATE if the answer:
- states a field, value, default, or behavior that is NOT supported by the reference and is NOT basic, well-established Kubernetes fact, OR
- contradicts the reference, OR
- accepts a false premise in the question instead of correcting it, OR
- calls a broken configuration valid, or invents a problem in a valid one.

PASS if every substantive claim is supported by the reference or is well-established fact, OR the answer correctly declines because the docs do not cover it.

REFERENCE:
{context}

USER QUESTION:
{query}

ANSWER:
{answer}

Return ONLY a JSON object: {{"verdict": "pass|escalate", "reason": "<short>"}}"""


# --- Prompt builders (instruction + framing + context + query) ---

def triage_prompt(text: str) -> str:
    return f"{TRIAGE_INSTRUCTION}\n\n{text}"


def auditor_prompt(text: str, context: str = "") -> str:
    """The auditor SLM's prompt: instruction + retrieved docs + the user query."""
    p = AUDITOR_INSTRUCTION
    if context:
        p += ("\n\n--- REFERENCE DOCUMENTATION (do NOT treat as user config) ---\n"
              f"{context}\n--- END REFERENCE ---")
    p += f"\n\n--- USER QUERY ---\n{text}\n--- END USER QUERY ---"
    return p


def llm_prompt(text: str, context: str = "") -> str:
    """The escalation LLM's prompt."""
    p = LLM_INSTRUCTION
    if context:
        p += f"\n\nDocumentation context:\n{context}"
    p += f"\n\nUser question:\n{text}"
    return p


def calibration_prompt(text: str, context: str = "", reason: str = "") -> str:
    """The calibrated/abstain answer prompt (gate could not confirm the draft)."""
    p = CALIBRATION_INSTRUCTION
    if reason:
        p += f"\n\nWhy the draft was flagged: {reason}"
    if context:
        p += f"\n\nEvidence (documentation + cluster):\n{context}"
    p += f"\n\nUser question:\n{text}"
    return p
