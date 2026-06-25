"""Intent-routed orchestration (LangGraph).

Doc-first by default. Tools are opt-in:
  - the user EXPLICITLY asks to act on the cluster (intent == "action")  -> run tools, then answer
  - otherwise answer from the KB; if the faithfulness gate flags a RUNTIME claim
    the docs can't confirm, gather live evidence (autopilot) or PROPOSE it and
    wait for the user (supervised).

  triage -> intent ─┬─ action -> gather(cluster tools) -> retrieve -> answer -> gate
                    └─ answer -> lint(if manifest)      -> retrieve -> answer -> gate
  gate: accept | refine(deprecated fix) | verify(runtime claim) | escalate(LLM)

Each node streams the SSE vocabulary the UI consumes (step_start/step_done/
model/token/answer_reset/response).
"""
import asyncio
import time
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from . import classify, config, extract, gate, generation, retrieval, tools
from . import remediation
from .validation import validate_config, validate_draft_fix


class AgentState(TypedDict, total=False):
    query: str
    autopilot: bool
    category: str
    confidence: str
    intent: str
    pending_tools: list
    tool_results: list
    evidence: list
    docs: list
    broaden: bool
    draft: str
    attempts: int
    critic_pass: bool
    correction: str
    needs_evidence: bool
    gate_reason: str
    used_llm: bool


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


def _build_context(state: AgentState) -> str:
    """Grounding = live cluster evidence + deterministic lint + retrieved docs."""
    sections = []
    if state.get("tool_results"):
        sections.append("LIVE CLUSTER EVIDENCE (read-only tools):\n" + tools.format_tool_results(state["tool_results"]))
    if state.get("evidence"):
        sections.append("VALIDATION (client-side checks):\n" + "\n".join(state["evidence"]))
    if state.get("docs"):
        sections.append("\n\n---\n\n".join(state["docs"]))
    return "\n\n===\n\n".join(sections)


async def _stream_answer(writer, name: str, token_stream) -> str:
    """Relay a model's token stream onto the SSE vocabulary the UI consumes, and
    return the full text. Emits the step_done timing on the first token (so the
    UI can show time-to-first-token) and a token event per chunk. Shared by every
    node that streams an answer (auditor SLM, LLM escalation, calibrated fallback)
    so the streaming contract lives in one place."""
    t = time.perf_counter()
    parts, first = [], True
    async for token in token_stream:
        if first:
            writer({"type": "step_done", "name": name, "ms": _ms(t), "detail": "time to first token"})
            first = False
        parts.append(token)
        writer({"type": "token", "text": token})
    if first:
        writer({"type": "step_done", "name": name, "ms": _ms(t), "detail": "no output"})
    return "".join(parts)


# --- Nodes ---

async def n_triage(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    writer({"type": "step_start", "name": "Triage classifier \u00b7 ONNX Runtime (CPU)", "note": "classifying intent"})
    t = time.perf_counter()
    result = await loop.run_in_executor(None, classify.classify, state["query"])
    cat = result["category"].replace("_", " ")
    detail = (f"off-topic \u00b7 {result['confidence']} confidence \u2192 reject"
              if result["category"] == "noise"
              else f"{cat} \u00b7 {result['confidence']} confidence \u2192 in-domain")
    writer({"type": "step_done", "name": "Triage classifier \u00b7 ONNX Runtime (CPU)", "ms": _ms(t), "detail": detail})
    return {"category": result["category"], "confidence": result["confidence"]}


async def n_reject(state: AgentState) -> dict:
    get_stream_writer()({"type": "response",
                         "text": "This does not look like a K8s autoscaling question."})
    return {}


async def n_intent(state: AgentState) -> dict:
    """Did the user explicitly ask to act on the live cluster? If so (and tools
    are available), queue the cluster tools; otherwise stay doc-first."""
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    writer({"type": "step_start", "name": "Intent router (CPU)", "note": "answer from docs, or act on the cluster?"})
    t = time.perf_counter()
    intent = await loop.run_in_executor(None, classify.classify_intent, state["query"])
    use_tools = intent == "action" and tools.available()
    pending = extract.select_cluster_tools(state["query"]) if use_tools else []
    detail = ("explicit cluster request \u2192 " + ", ".join(pending)) if pending else "answer from documentation"
    writer({"type": "step_done", "name": "Intent router (CPU)", "ms": _ms(t), "detail": detail})
    return {"intent": intent, "pending_tools": pending}


async def n_gather(state: AgentState) -> dict:
    """Run the queued read-only cluster tools and collect their evidence."""
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    results = list(state.get("tool_results", []))
    for tool in state.get("pending_tools", []):
        if len(results) >= config.MAX_TOOL_CALLS:
            break
        args = extract.extract_args(state["query"], tool)
        writer({"type": "step_start", "name": f"Tool \u00b7 {tool}", "note": extract.args_summary(tool, args)})
        t = time.perf_counter()
        output = await loop.run_in_executor(None, tools.run_tool, tool, args)
        writer({"type": "step_done", "name": f"Tool \u00b7 {tool}", "ms": _ms(t), "detail": tools.tool_detail(output)})
        results.append({"tool": tool, "args": args, "output": output})
    return {"tool_results": results, "pending_tools": []}


async def n_lint(state: AgentState) -> dict:
    """Doc-first path: if the user pasted a manifest, lint it (no cluster)."""
    if not extract.looks_like_yaml(state["query"]):
        return {}
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    writer({"type": "step_start", "name": "Config validator (CPU)", "note": "linting pasted manifest"})
    t = time.perf_counter()
    result = await loop.run_in_executor(None, validate_config, {"yaml": extract.extract_manifest(state["query"])})
    writer({"type": "step_done", "name": "Config validator (CPU)", "ms": _ms(t), "detail": result})
    return {"evidence": [f"[validate_config] {result}"]}


async def n_retrieve(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    query = state["query"]
    broaden = state.get("broaden", False)
    writer({"type": "step_start", "name": "Retriever (tuned encoder, CPU)", "note": "embedding query \u2192 768d"})
    t = time.perf_counter()
    embedding = await loop.run_in_executor(None, retrieval.embed_query, query)
    writer({"type": "step_done", "name": "Retriever (tuned encoder, CPU)", "ms": _ms(t), "detail": "domain-tuned encoder"})

    writer({"type": "step_start", "name": "OpenSearch (vector DB)", "note": "hybrid k-NN + BM25"})
    t = time.perf_counter()
    candidates = await loop.run_in_executor(None, retrieval.hybrid_candidates, embedding, query, broaden)
    writer({"type": "step_done", "name": "OpenSearch (vector DB)", "ms": _ms(t), "detail": f"{len(candidates)} candidates"})

    keep = config.KEEP_DOCS + (config.BROADEN_EXTRA if broaden else 0)
    writer({"type": "step_start", "name": "Reranker (cross-encoder, CPU)", "note": f"scoring {len(candidates)} \u2192 top {keep}"})
    t = time.perf_counter()
    docs = await loop.run_in_executor(None, retrieval.rerank_docs, query[:config.RERANK_QUERY_CHARS], candidates, keep)
    writer({"type": "step_done", "name": "Reranker (cross-encoder, CPU)", "ms": _ms(t), "detail": f"kept top {len(docs)}"})
    return {"docs": docs}


async def n_generate(state: AgentState) -> dict:
    writer = get_stream_writer()
    context = _build_context(state)
    if config.DEBUG_CONTEXT:
        writer({"type": "debug_context", "tools": [r.get("tool") for r in state.get("tool_results", [])],
                "doc_count": len(state.get("docs", [])), "context": context})
    if state.get("correction"):
        context += "\n\n=== CORRECTION REQUIRED ===\n" + state["correction"]
    attempts = state.get("attempts", 0)
    if attempts > 0:
        writer({"type": "answer_reset", "reason": "refining"})

    unclassified = state.get("category", "unknown") in (None, "unknown")
    if unclassified:
        name, stream_fn, used_llm = "LLM API (Bedrock fallback)", generation.stream_llm, True
        writer({"type": "model", "name": "Claude Sonnet 4.5 (Bedrock)"})
    else:
        name, stream_fn, used_llm = "Auditor SLM (8B, CPU)", generation.stream_slm, False
        writer({"type": "model", "name": "Auditor SLM (8B, CPU)"})

    writer({"type": "step_start", "name": name, "note": "generating answer"})
    draft = await _stream_answer(writer, name, stream_fn(state["query"], context))
    return {"draft": draft, "attempts": attempts + 1, "used_llm": used_llm}


async def n_critic(state: AgentState) -> dict:
    """Faithfulness gate (LLM) + deterministic deprecated-config lint. Decides:
    accept, retry a deprecated fix, gather live evidence for a runtime claim, or
    escalate to the LLM."""
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    draft = state.get("draft", "")
    context = _build_context(state)
    used_llm = state.get("used_llm", False)
    attempts = state.get("attempts", 0)
    writer({"type": "step_start", "name": "Faithfulness gate (LLM)", "note": "is the draft supported by the evidence?"})
    t = time.perf_counter()
    fix_issues = await loop.run_in_executor(None, validate_draft_fix, draft)
    # Gate every answer, including the escalated LLM's: the top of the ladder is
    # not exempt. If even the LLM answer isn't supported, we abstain rather than
    # ship a confidently-wrong answer.
    escalate, reason = await loop.run_in_executor(None, gate.llm_gate, state["query"], draft, context)

    passed = (not escalate) and (not fix_issues)
    can_retry = attempts <= config.MAX_CRITIC_RETRIES
    # A flagged answer about live state, on a query we haven't yet inspected the
    # cluster for, calls for real evidence rather than escalation/speculation.
    needs_evidence = (escalate and not used_llm and not fix_issues and tools.available()
                      and extract.is_operational(state["query"]) and not state.get("tool_results"))

    if passed:
        verdict = "accepted"
    elif fix_issues and can_retry:
        verdict = "refining (deprecated config)"
    elif needs_evidence:
        verdict = "needs live evidence"
    elif used_llm:
        verdict = "could not confirm \u2014 calibrating an honest answer"
    else:
        verdict = "escalating to LLM"
    detail = "supported" if not escalate else f"not supported \u2192 {'abstain' if used_llm else 'escalate'}: {reason}"
    if fix_issues:
        detail += " \u00b7 fix uses deprecated/invalid config"
    writer({"type": "step_done", "name": "Faithfulness gate (LLM)", "ms": _ms(t), "detail": f"{detail} \u00b7 {verdict}"})

    out = {"critic_pass": passed, "correction": "", "needs_evidence": needs_evidence, "gate_reason": reason}
    if fix_issues and can_retry:
        out["correction"] = ("Your previous draft proposed deprecated or invalid configuration: "
                             + "; ".join(fix_issues)
                             + ". Re-issue the fix using only current, non-deprecated APIs from the documentation.")
    # Autopilot self-verifies automatically; supervised mode proposes (n_propose).
    if needs_evidence and state.get("autopilot"):
        out["pending_tools"] = ["investigate_cluster"]
    return out


async def n_propose(state: AgentState) -> dict:
    """Supervised mode: the answer needs live evidence we won't gather without
    consent. Offer the read-only check rather than speculate."""
    get_stream_writer()({"type": "response", "text": (
        "I answered from the documentation, but I can't fully confirm this against your "
        "actual cluster without inspecting it. Want me to run a read-only check of the "
        "relevant resources to verify?")})
    return {}


async def n_escalate(state: AgentState) -> dict:
    writer = get_stream_writer()
    context = _build_context(state)
    writer({"type": "answer_reset", "reason": "escalating"})
    writer({"type": "model", "name": "Claude Sonnet 4.5 (Bedrock)"})
    writer({"type": "step_start", "name": "LLM API (Bedrock escalation)", "note": "CPU answer not supported \u2014 escalating"})
    await _stream_answer(writer, "LLM API (Bedrock escalation)", generation.stream_llm(state["query"], context))
    return {"used_llm": True}


async def n_abstain(state: AgentState) -> dict:
    """The escalated LLM answer also failed the gate — there is no higher model to
    escalate to. Instead of shipping an unsupported answer, produce a calibrated,
    abstention-aware reply: state only what the evidence supports and say plainly
    what could not be confirmed. This is the "never confidently wrong" backstop."""
    writer = get_stream_writer()
    context = _build_context(state)
    writer({"type": "answer_reset", "reason": "calibrating"})
    writer({"type": "model", "name": "Claude Sonnet 4.5 (Bedrock)"})
    writer({"type": "step_start", "name": "Calibrated answer (LLM)",
            "note": "evidence did not fully support the draft \u2014 answering with calibrated confidence"})
    await _stream_answer(writer, "Calibrated answer (LLM)",
                         generation.stream_calibrated(state["query"], context, state.get("gate_reason", "")))
    return {}


async def n_remediate(state: AgentState) -> dict:
    """After answering: if a safe, bounded fix applies to a resource the user
    named, apply it (autopilot) or propose it for one-click apply (supervised).

    The apply itself dry-runs server-side first and re-reads to verify. Read-only
    by default — no mutation unless autopilot AND ALLOW_APPLY AND a whitelisted
    remediation is detected for a specific named target.

    (Planned extension: when a proposed fix targets a resource that does NOT
    exist in the cluster, offer a confirm-gated server-side dry-run to validate
    the fix without persisting it.)
    """
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    rem = await loop.run_in_executor(None, remediation.detect_remediation, state["query"])
    if not rem:
        return {}
    if not state.get("autopilot"):
        writer({"type": "response", "text": (
            "**Autopilot is off, so I won't change anything in the cluster.** "
            "Here's the fix I'd apply \u2014 click **Apply this fix**, or apply it yourself below.")})
        writer({"type": "proposal", "action": rem["action"], "target": rem["target"],
                "summary": rem["summary"], "manual": rem.get("manual", "")})
        return {}
    apply_fn, verify_fn = remediation.REMEDIATIONS[rem["action"]]
    target = rem["target"]
    writer({"type": "response", "text": (
        f"**Autopilot is on \u2014 applying now.** Bounded, whitelisted change "
        f"({rem['summary']}); it dry-runs first, then I re-read `{target}` to verify.")})
    writer({"type": "step_start", "name": "Apply fix (autopilot)", "note": rem["summary"]})
    t = time.perf_counter()
    result = await loop.run_in_executor(None, apply_fn, target)
    writer({"type": "step_done", "name": "Apply fix (autopilot)", "ms": _ms(t), "detail": result["message"]})
    if not result["ok"]:
        writer({"type": "response", "text": f"**Autopilot could not apply the fix:** {result['message']}"})
        return {}
    writer({"type": "step_start", "name": "Verify (CPU)", "note": f"re-checking {target}"})
    t = time.perf_counter()
    check = await loop.run_in_executor(None, verify_fn, target)
    writer({"type": "step_done", "name": "Verify (CPU)", "ms": _ms(t), "detail": check["message"]})
    status = "applied and verified" if check["ok"] else "applied, but verification failed"
    writer({"type": "response", "text": f"**Autopilot {status}.** {check['message']}"})
    return {}


# --- Routing ---

def _route_after_triage(state: AgentState) -> str:
    return "reject" if state.get("category") == "noise" else "intent"


def _route_after_intent(state: AgentState) -> str:
    return "gather" if state.get("pending_tools") else "lint"


def _route_after_critic(state: AgentState) -> str:
    if state.get("critic_pass"):
        return "end"
    if state.get("correction") and state.get("attempts", 0) <= config.MAX_CRITIC_RETRIES:
        return "refine"
    if state.get("needs_evidence"):
        return "verify" if state.get("autopilot") else "propose"
    # The LLM answer also failed the gate: no higher model — abstain honestly.
    if state.get("used_llm"):
        return "abstain"
    return "escalate"


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("triage", n_triage)
    g.add_node("reject", n_reject)
    g.add_node("intent", n_intent)
    g.add_node("gather", n_gather)
    g.add_node("lint", n_lint)
    g.add_node("retrieve", n_retrieve)
    g.add_node("generate", n_generate)
    g.add_node("critic", n_critic)
    g.add_node("propose", n_propose)
    g.add_node("escalate", n_escalate)
    g.add_node("abstain", n_abstain)
    g.add_node("remediate", n_remediate)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", _route_after_triage, {"reject": "reject", "intent": "intent"})
    g.add_edge("reject", END)
    g.add_conditional_edges("intent", _route_after_intent, {"gather": "gather", "lint": "lint"})
    g.add_edge("gather", "retrieve")
    g.add_edge("lint", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "critic")
    g.add_conditional_edges("critic", _route_after_critic,
                            {"end": "remediate", "refine": "generate", "verify": "gather",
                             "propose": "propose", "escalate": "escalate", "abstain": "abstain"})
    g.add_edge("escalate", "critic")
    g.add_edge("abstain", END)
    g.add_edge("propose", END)
    g.add_edge("remediate", END)
    return g.compile()


agent = build_agent()
