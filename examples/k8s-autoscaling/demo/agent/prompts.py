"""All prompt text for the agent, in one place.

Every instruction string and prompt template the agent sends to a model lives
here, with small builder functions that assemble the final prompt (instruction +
framing + context + query). Centralized so the prompts can be read, reviewed, and
edited together rather than scattered across modules. The one exception is the
domain text of the triage prompt: it is read from project.domain in
triage/expert.yaml, the same rules the classifier's training data was written
against, so the two triage models follow one definition of each label.

Which model sees which prompt:
  - triage_llm_prompt -> frontier model (TRIAGE=llm) : classify intent; the classifier gets the raw message
  - analyst_prompt  -> analyst SLM (llama.cpp, CPU)  : the grounded draft answer
  - llm_prompt      -> escalation LLM (Bedrock)      : when the gate escalates
  - calibration_prompt -> calibration LLM (Bedrock)  : the abstain backstop
  - GATE_PROMPT     -> faithfulness gate LLM (Bedrock): is the draft supported?
  - fix_proposal_prompt -> analyst SLM (llama.cpp, CPU), schema-constrained:
    propose a remediation within patch_schema's allowed fields
"""
import functools
import os

import yaml

from . import patch_schema

# --- The triage domain (from triage/expert.yaml) ---

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# The repo checkout (make triage-acc on the IDE), then the copy the image
# build stages next to this file (the image build context is demo/ only).
_TRIAGE_EXPERT_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(_AGENT_DIR)), "triage", "expert.yaml"),
    os.path.join(_AGENT_DIR, "triage_expert.yaml"),
)


def triage_expert_path() -> str:
    """The triage expert config: env TRIAGE_EXPERT_CONFIG when set, else the
    repo's triage/expert.yaml, else agent/triage_expert.yaml in the image."""
    env = os.environ.get("TRIAGE_EXPERT_CONFIG", "")
    tried = [env] if env else list(_TRIAGE_EXPERT_CANDIDATES)
    for path in tried:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "TRIAGE=llm needs the triage expert config for its prompt; tried "
        + ", ".join(tried) + ". Set TRIAGE_EXPERT_CONFIG to the path of triage/expert.yaml.")


@functools.lru_cache(maxsize=1)
def triage_domain() -> tuple[tuple[str, ...], str]:
    """(labels.routing, project.domain) from the triage expert config."""
    path = triage_expert_path()
    with open(path, encoding="utf-8") as f:
        project = (yaml.safe_load(f) or {}).get("project") or {}
    labels = tuple(((project.get("labels") or {}).get("routing")) or ())
    domain = " ".join(str(project.get("domain") or "").split())
    if not labels or not domain:
        raise ValueError(f"{path}: project.domain and project.labels.routing must both be set")
    return labels, domain


# --- Instruction strings ---

# The frontier model's zero-shot triage prompt (TRIAGE=llm): the label names,
# the labeling rules from project.domain, and the one-line output the parser
# reads. The classifier does not see this prompt; it is sent the message
# alone, which is what it was trained on.
TRIAGE_LLM_FORMAT = (
    "Reply with one line and nothing else, in the form  label|confidence  where "
    "confidence is high, medium, or low. Example:  karpenter_config|high"
)


def triage_llm_instruction() -> str:
    labels, domain = triage_domain()
    return ("Classify the support message below into exactly one routing category. "
            "The categories:\n"
            + "".join(f"  {label}\n" for label in labels)
            + f"\nLabeling rules:\n{domain}\n\n{TRIAGE_LLM_FORMAT}")

ANALYST_INSTRUCTION = (
    "You are a Kubernetes autoscaling auditor. "
    "Answer ONLY based on the reference documentation below. "
    "Do NOT invent fields, behaviors, or modes not in the docs. "
    "If the docs don't cover something, say so. "
    "State what is correct, why, and provide a fix if needed."
)

LLM_INSTRUCTION = (
    "You are a Kubernetes autoscaling expert. Answer the user's question accurately "
    "using the provided documentation context. Be specific and include YAML examples "
    "when relevant. If the documentation doesn't cover the topic, say so. If the "
    "question is not about Kubernetes autoscaling at all, say briefly that it is out "
    "of scope and do not answer it."
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
- calls a broken configuration valid, or invents a problem in a valid one, OR
- is INCOMPLETE on an enumeration: the question asks which values/options/policies are valid (or to list them), the reference enumerates a definite set, and the answer omits one or more of those reference-listed items. A confidently-worded partial list is still wrong, OR
- asserts a closed set ("the only values are...", "no other values are valid", "there are exactly two") that leaves out an item the reference lists.

Judge completeness only against what the reference actually enumerates: do not escalate for leaving out an item the reference never mentions, and do not demand detail the question did not ask for. An answer that correctly lists every reference-listed item is complete even if terse.

PASS if every substantive claim is supported by the reference or is well-established fact AND, for an enumeration question, the answer covers every item the reference lists; OR the answer correctly declines because the docs do not cover it.

REFERENCE:
{context}

USER QUESTION:
{query}

ANSWER:
{answer}

Return ONLY a JSON object: {{"verdict": "pass|escalate", "reason": "<short>"}}"""


# --- Prompt builders (instruction + framing + context + query) ---

def triage_llm_prompt(text: str) -> str:
    """The frontier model's zero-shot triage prompt (TRIAGE=llm only; the
    classifier is sent the raw message, see classify.py)."""
    return f"{triage_llm_instruction()}\n\n--- MESSAGE ---\n{text}\n--- END MESSAGE ---\n\nAnswer:"


def analyst_prompt(text: str, context: str = "") -> str:
    """The analyst SLM's prompt: instruction + retrieved docs + the user query."""
    p = ANALYST_INSTRUCTION
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


FIX_PROPOSAL_INSTRUCTION = (
    "You are a Kubernetes remediation planner. You will be shown a diagnosed problem "
    "with one named resource, plus the ONLY fields you are allowed to change on that "
    "resource (nothing else may ever be changed). Propose exactly one field+value fix "
    "that addresses the diagnosis, or say no safe fix applies. "
    "Only use a field from the allowed list. If the diagnosis does not match any "
    "allowed field, set no_fix to true rather than forcing an unrelated change."
)


def fix_proposal_prompt(kind: str, target: str, diagnosis: str) -> str:
    """The analyst SLM's remediation-proposal prompt: instruction + the allowed
    fields for this kind (from patch_schema) + the diagnosed problem. The
    response is constrained to patch_schema.json_schema_for_kind(kind), so the
    model cannot name a field outside this list even if it tried."""
    fields = patch_schema.fields_for_kind(kind)
    allowed = "\n".join(f"- {f}: {d}" for f, d in fields.items()) or "(no fields are allowed for this kind)"
    return (
        f"{FIX_PROPOSAL_INSTRUCTION}\n\n"
        f"--- RESOURCE ---\n{kind} {target}\n\n"
        f"--- ALLOWED FIELDS (only these may ever be changed) ---\n{allowed}\n\n"
        f"--- DIAGNOSIS ---\n{diagnosis}\n"
    )


def calibration_prompt(text: str, context: str = "", reason: str = "") -> str:
    """The calibrated/abstain answer prompt (gate could not confirm the draft)."""
    p = CALIBRATION_INSTRUCTION
    if reason:
        p += f"\n\nWhy the draft was flagged: {reason}"
    if context:
        p += f"\n\nEvidence (documentation + cluster):\n{context}"
    p += f"\n\nUser question:\n{text}"
    return p
