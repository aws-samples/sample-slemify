"""Intent-routed orchestration (LangGraph).

Doc-first by default. Tools are opt-in:
  - the user EXPLICITLY asks to act on the cluster (intent == "action")  -> run tools, then answer
  - otherwise answer from the KB; if the faithfulness gate flags a RUNTIME claim
    the docs can't confirm, gather live evidence (autopilot) or PROPOSE it and
    wait for the user (supervised).

  triage -> intent ─┬─ action -> gather(cluster tools) -> retrieve -> answer -> gate
                    └─ answer -> lint(if manifest)      -> retrieve -> answer -> gate
  gate: accept | refine(deprecated fix) | verify(runtime claim) | escalate(LLM)

Seats can be off (config.py). TRIAGE=off removes triage, intent, tools, and the
lint: the query goes straight to retrieval. GATE=off ships the draft as is.
Both off with EMBED=bedrock and ANALYST=llm is the monolith: embed, search,
one frontier-model call.

Each node streams the SSE vocabulary the UI consumes (step_start/step_done/
model/token/answer_reset/response). Step names say who actually filled each
seat (config.py: TRIAGE, EMBED, RERANK, ANALYST, GATE), so the UI, the logs, and the
eval describe the configuration that ran, not the one the code assumed.
"""
import asyncio
import time
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from . import classify, config, extract, gate, generation, metrics, patch_schema, prompts, retrieval, tools
from . import toolclient
from .validation import validate_config, validate_draft_fix

# --- Seat labels (what the UI and eval see) ---
LBL_TRIAGE = ("Triage \u00b7 LLM (Bedrock)" if config.TRIAGE == "llm"
              else "Triage classifier \u00b7 ONNX Runtime (CPU)")
LBL_INTENT = "Intent router (LLM)" if config.TRIAGE == "llm" else "Intent router (CPU)"
LBL_EMBED = ("Retriever (Bedrock embeddings)" if config.EMBED == "bedrock"
             else "Retriever (tuned encoder, CPU)")
LBL_EMBED_DETAIL = (f"Titan, {config.BEDROCK_EMBED_DIM}d" if config.EMBED == "bedrock"
                    else "domain-tuned encoder, 768d")
LBL_RERANK = "Reranker (off)" if config.RERANK == "off" else "Reranker (cross-encoder, CPU)"
LBL_ANALYST = "Analyst \u00b7 LLM (Bedrock)" if config.ANALYST == "llm" else "Analyst SLM (CPU)"
MODEL_ANALYST = "LLM (Bedrock)" if config.ANALYST == "llm" else "Analyst SLM (CPU)"


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
    node that streams an answer (analyst SLM, LLM escalation, calibrated fallback)
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
    writer({"type": "step_start", "name": LBL_TRIAGE, "note": "classifying intent"})
    t = time.perf_counter()
    result = await asyncio.to_thread(classify.classify, state["query"])
    cat = result["category"].replace("_", " ")
    detail = (f"off-topic \u00b7 {result['confidence']} confidence \u2192 reject"
              if result["category"] == "noise"
              else f"{cat} \u00b7 {result['confidence']} confidence \u2192 in-domain")
    writer({"type": "step_done", "name": LBL_TRIAGE, "ms": _ms(t), "detail": detail})
    return {"category": result["category"], "confidence": result["confidence"]}


async def n_reject(state: AgentState) -> dict:
    get_stream_writer()({"type": "response",
                         "text": "This does not look like a K8s autoscaling question."})
    return {}


async def n_intent(state: AgentState) -> dict:
    """Did the user explicitly ask to act on the live cluster? If so (and tools
    are available), queue the cluster tools; otherwise stay doc-first."""
    writer = get_stream_writer()
    writer({"type": "step_start", "name": LBL_INTENT, "note": "answer from docs, or act on the cluster?"})
    t = time.perf_counter()
    intent = await asyncio.to_thread(classify.classify_intent, state["query"])
    use_tools = intent == "action" and toolclient.available()
    pending = extract.select_cluster_tools(state["query"]) if use_tools else []
    detail = ("explicit cluster request \u2192 " + ", ".join(pending)) if pending else "answer from documentation"
    writer({"type": "step_done", "name": LBL_INTENT, "ms": _ms(t), "detail": detail})
    return {"intent": intent, "pending_tools": pending}


async def n_gather(state: AgentState) -> dict:
    """Run the queued read-only cluster tools and collect their evidence."""
    writer = get_stream_writer()
    results = list(state.get("tool_results", []))
    for tool in state.get("pending_tools", []):
        if len(results) >= config.MAX_TOOL_CALLS:
            break
        args = extract.extract_args(state["query"], tool)
        writer({"type": "step_start", "name": f"Tool \u00b7 {tool}", "note": extract.args_summary(tool, args)})
        t = time.perf_counter()
        output = await asyncio.to_thread(toolclient.run_tool, tool, args)
        writer({"type": "step_done", "name": f"Tool \u00b7 {tool}", "ms": _ms(t), "detail": tools.tool_detail(output)})
        results.append({"tool": tool, "args": args, "output": output})
    return {"tool_results": results, "pending_tools": []}


async def n_lint(state: AgentState) -> dict:
    """Doc-first path: if the user pasted a manifest, lint it (no cluster)."""
    if not extract.looks_like_yaml(state["query"]):
        return {}
    writer = get_stream_writer()
    writer({"type": "step_start", "name": "Config validator (CPU)", "note": "linting pasted manifest"})
    t = time.perf_counter()
    result = await asyncio.to_thread(validate_config, {"yaml": extract.extract_manifest(state["query"])})
    writer({"type": "step_done", "name": "Config validator (CPU)", "ms": _ms(t), "detail": result})
    return {"evidence": [f"[validate_config] {result}"]}


async def n_retrieve(state: AgentState) -> dict:
    writer = get_stream_writer()
    query = state["query"]
    broaden = state.get("broaden", False)
    writer({"type": "step_start", "name": LBL_EMBED, "note": "embedding query"})
    t = time.perf_counter()
    embedding = await asyncio.to_thread(retrieval.embed_query, query)
    writer({"type": "step_done", "name": LBL_EMBED, "ms": _ms(t), "detail": LBL_EMBED_DETAIL})

    writer({"type": "step_start", "name": "OpenSearch (vector DB)", "note": "hybrid k-NN + BM25"})
    t = time.perf_counter()
    candidates = await asyncio.to_thread(retrieval.hybrid_candidates, embedding, query, broaden)
    writer({"type": "step_done", "name": "OpenSearch (vector DB)", "ms": _ms(t), "detail": f"{len(candidates)} candidates"})

    keep = config.KEEP_DOCS + (config.BROADEN_EXTRA if broaden else 0)
    note = (f"keeping top {keep} in vector order" if config.RERANK == "off"
            else f"scoring {len(candidates)} \u2192 top {keep}")
    writer({"type": "step_start", "name": LBL_RERANK, "note": note})
    t = time.perf_counter()
    docs = await asyncio.to_thread(retrieval.rerank_docs, query[:config.RERANK_QUERY_CHARS], candidates, keep)
    writer({"type": "step_done", "name": LBL_RERANK, "ms": _ms(t), "detail": f"kept top {len(docs)}"})

    # Field-aware guarantee for pasted configs: similarity ranking can bury the
    # one chunk that DEFINES a field the manifest uses (the model then
    # fabricates about exactly that field — proven failure mode). Fetch each
    # used field's authoritative definition and append what the reranked set
    # is missing. Additive, so the similarity-ranked grounding is untouched.
    fields = extract.manifest_fields(query)
    if fields:
        writer({"type": "step_start", "name": "Field definitions (OpenSearch)",
                "note": f"authoritative docs for: {', '.join(fields[:6])}"})
        t = time.perf_counter()
        extra = await asyncio.to_thread(retrieval.field_definition_docs, fields, docs)
        writer({"type": "step_done", "name": "Field definitions (OpenSearch)", "ms": _ms(t),
                "detail": f"added {len(extra)} definition chunk(s)" if extra else "already covered"})
        docs = docs + extra
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

    # With no triage seat there is no category, and that is not a failure to
    # classify: the SLM drafts every query.
    unclassified = config.TRIAGE != "off" and state.get("category", "unknown") in (None, "unknown")
    # Who drafts: the ANALYST seat. With the LLM in the seat (the monolith, or
    # the one-variable control: same graph, context, gate, and judge, only the
    # drafter changed) every query goes to Bedrock. With the SLM in the seat,
    # only an unclassifiable query falls back to the LLM.
    if config.ANALYST == "llm":
        name, stream_fn, used_llm = LBL_ANALYST, generation.stream_llm, True
        writer({"type": "model", "name": MODEL_ANALYST})
    elif unclassified:
        name, stream_fn, used_llm = "LLM API (Bedrock fallback)", generation.stream_llm, True
        writer({"type": "model", "name": "LLM (Bedrock)"})
    else:
        name, stream_fn, used_llm = LBL_ANALYST, generation.stream_slm, False
        writer({"type": "model", "name": MODEL_ANALYST})

    writer({"type": "step_start", "name": name, "note": "generating answer"})
    draft = await _stream_answer(writer, name, stream_fn(state["query"], context))
    return {"draft": draft, "attempts": attempts + 1, "used_llm": used_llm}


async def n_critic(state: AgentState) -> dict:
    """Faithfulness gate (LLM) + deterministic deprecated-config lint. Decides:
    accept, retry a deprecated fix, gather live evidence for a runtime claim, or
    escalate to the LLM."""
    writer = get_stream_writer()
    draft = state.get("draft", "")
    context = _build_context(state)
    used_llm = state.get("used_llm", False)
    attempts = state.get("attempts", 0)
    writer({"type": "step_start", "name": "Faithfulness gate (LLM)", "note": "is the draft supported by the evidence?"})
    t = time.perf_counter()
    fix_issues = await asyncio.to_thread(validate_draft_fix, draft)
    # Gate every answer, including the escalated LLM's: the top of the ladder is
    # not exempt. If even the LLM answer isn't supported, we abstain rather than
    # ship a confidently-wrong answer.
    escalate, reason = await asyncio.to_thread(gate.llm_gate, state["query"], draft, context)

    passed = (not escalate) and (not fix_issues)
    can_retry = attempts <= config.MAX_CRITIC_RETRIES
    # A flagged answer about live state, on a query we haven't yet inspected the
    # cluster for, calls for real evidence rather than escalation/speculation.
    needs_evidence = (escalate and not used_llm and not fix_issues and toolclient.available()
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

    # Instrument every gate check, and the terminal resolutions, so the real
    # production pass rate (not the eval proxy) is measurable. The economics of
    # this architecture hinge on the SLM's first-pass rate; see agent/metrics.py.
    category = state.get("category", "unknown")
    metrics.record_gate_check(category, used_llm, escalate, attempts, reason or "")
    if passed:
        metrics.record_outcome(category, "llm_pass" if used_llm else "slm_pass")
    elif used_llm and not (fix_issues and can_retry) and not needs_evidence:
        metrics.record_outcome(category, "abstain")
    elif needs_evidence and not state.get("autopilot"):
        metrics.record_outcome(category, "propose")
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
    writer({"type": "model", "name": "LLM (Bedrock)"})
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
    writer({"type": "model", "name": "LLM (Bedrock)"})
    writer({"type": "step_start", "name": "Calibrated answer (LLM)",
            "note": "evidence did not fully support the draft \u2014 answering with calibrated confidence"})
    await _stream_answer(writer, "Calibrated answer (LLM)",
                         generation.stream_calibrated(state["query"], context, state.get("gate_reason", "")))
    return {}


async def _find_fix(state: AgentState, writer) -> dict | None:
    """Two ways to find a fix, in order:
      1. detect_remediation: cheap, evidence-checked heuristics for the known
         problem shapes this demo ships scenarios for (no model call).
      2. Otherwise, if the query names a resource of a kind PATCH_SCHEMA covers
         at all, ask the analyst SLM (CPU) to propose a fix -- constrained by
         response_format to only ever name a field from patch_schema for this
         kind. Its proposal is then re-validated against the schema (defense
         in depth: the grammar should already guarantee this) before it is
         shown to anyone or ever reaches apply.
    Either way the result is the same shape: {kind, target, field, value,
    summary, manual}. Returns None if no fix is found or a proposal fails to
    validate -- never a best-effort guess.
    """
    rem = await asyncio.to_thread(toolclient.detect_remediation, state["query"])
    if rem:
        return rem

    target_info = await asyncio.to_thread(toolclient.named_target, state["query"])
    if not target_info:
        return None
    kind, target = target_info["kind"], target_info["target"]
    diagnosis = state.get("draft", "")
    prompt = prompts.fix_proposal_prompt(kind, target, diagnosis)
    schema = patch_schema.json_schema_for_kind(kind)
    writer({"type": "step_start", "name": "Fix proposal (Analyst SLM, CPU)",
            "note": f"proposing a schema-constrained fix for {kind} {target}"})
    t = time.perf_counter()
    proposal = await generation.propose_fix(prompt, schema)
    if not proposal or proposal.get("no_fix") or proposal.get("field") in (None, "none"):
        writer({"type": "step_done", "name": "Fix proposal (Analyst SLM, CPU)", "ms": _ms(t),
                "detail": "no safe fix proposed"})
        return None
    field, value = proposal.get("field", ""), proposal.get("value", "")
    plan = await asyncio.to_thread(toolclient.plan_fix, kind, target, field, value)
    if not plan.get("ok"):
        writer({"type": "step_done", "name": "Fix proposal (Analyst SLM, CPU)", "ms": _ms(t),
                "detail": f"proposal rejected by schema: {plan.get('message')}"})
        return None
    writer({"type": "step_done", "name": "Fix proposal (Analyst SLM, CPU)", "ms": _ms(t),
            "detail": f"{field} -> {value}"})
    return {"kind": kind, "target": target, "field": field, "value": value,
            "summary": plan.get("message", patch_schema.describe_fix(kind, field, value)),
            "manual": patch_schema.manual_command(kind, plan.get("name", target), plan.get("namespace"),
                                                   plan.get("patch") or {}) if plan.get("patch") else ""}


async def n_remediate(state: AgentState) -> dict:
    """After answering: if a safe, bounded fix applies to a resource the user
    named, apply it (autopilot) or propose it for one-click apply (supervised).

    The apply itself dry-runs server-side first and re-reads to verify. Read-only
    by default — no mutation unless autopilot AND ALLOW_APPLY AND a fix is found
    (by the evidence-based heuristic, or a model proposal validated against
    patch_schema) for a specific named target. See PERMISSIONS.md for exactly
    what this can ever touch and why.
    """
    writer = get_stream_writer()
    rem = await _find_fix(state, writer)
    if not rem:
        return {}
    kind, target, field, value = rem["kind"], rem["target"], rem["field"], rem["value"]
    if not state.get("autopilot"):
        writer({"type": "response", "text": (
            "**Autopilot is off, so I won't change anything in the cluster.** "
            "Here's the fix I'd apply \u2014 click **Apply this fix**, or apply it yourself below.")})
        writer({"type": "proposal", "kind": kind, "target": target, "field": field, "value": value,
                "summary": rem["summary"], "manual": rem.get("manual", "")})
        return {}
    writer({"type": "response", "text": (
        f"**Autopilot is on \u2014 applying now.** Bounded, schema-checked change "
        f"({rem['summary']}); it dry-runs first, then I re-read `{target}` to verify.")})
    writer({"type": "step_start", "name": "Apply fix (autopilot)", "note": rem["summary"]})
    t = time.perf_counter()
    result = await asyncio.to_thread(toolclient.apply_fix, kind, target, field, value)
    writer({"type": "step_done", "name": "Apply fix (autopilot)", "ms": _ms(t), "detail": result["message"]})
    if not result["ok"]:
        writer({"type": "response", "text": f"**Autopilot could not apply the fix:** {result['message']}"})
        return {}
    writer({"type": "step_start", "name": "Verify (CPU)", "note": f"re-checking {target}"})
    t = time.perf_counter()
    check = await asyncio.to_thread(toolclient.verify_fix, kind, target, field, value)
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

    if config.TRIAGE == "off":
        # No triage seat: nothing decides category or intent, so there is no
        # reject path, no tools, and no manifest lint. Retrieval is the first step.
        g.add_edge(START, "retrieve")
    else:
        g.add_edge(START, "triage")
        g.add_conditional_edges("triage", _route_after_triage, {"reject": "reject", "intent": "intent"})
        g.add_edge("reject", END)
        g.add_conditional_edges("intent", _route_after_intent, {"gather": "gather", "lint": "lint"})
        g.add_edge("gather", "retrieve")
        g.add_edge("lint", "retrieve")
    g.add_edge("retrieve", "generate")
    if config.GATE == "off":
        # No gate seat: the draft ships as written. Nothing checks it, nothing
        # escalates, nothing remediates.
        g.add_edge("generate", END)
    else:
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
