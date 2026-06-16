"""Multi-agent orchestrator with RAG, read-only tools, and LLM fallback (LangGraph).

The control flow is a LangGraph state graph. After triage, a planner decides
whether the agent needs to gather live evidence with read-only tools
(describe a cluster resource, list its events, validate a pasted config) before
answering. It then retrieves docs, re-ranks them, and answers with the Auditor
SLM by default (or a Bedrock LLM fallback on low triage confidence). Each node
streams its progress (per-stage timing and tokens) to the chat UI via
LangGraph's custom stream writer, so the SSE contract the UI consumes is stable.

All tools are READ-ONLY: the agent talks to the Kubernetes API via the Python
client with structured, validated arguments (never shelled-out kubectl with
user text), and the orchestrator's RBAC grants only get/list/watch.

Usage:
  pip install fastapi uvicorn httpx opensearch-py boto3 langgraph kubernetes pyyaml
  python3 server.py
"""

import asyncio
import json
import os
import re
import time
from typing import TypedDict

import boto3
import httpx
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from kubernetes import client as k8s_client, config as k8s_config
from kubernetes.dynamic import DynamicClient
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from opensearchpy import OpenSearch
from pydantic import BaseModel

# --- Configuration ---

TRIAGE_URL = os.environ.get("TRIAGE_URL", "http://localhost:8081")
AUDITOR_URL = os.environ.get("AUDITOR_URL", "http://localhost:8082")
OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
LLM_MODEL = os.environ.get("LLM_MODEL", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
# In-cluster embedding served by the Slemify-trained retriever (task:
# embedding) over a TEI-compatible /embed endpoint. The domain-tuned encoder
# produces 768-dimensional vectors and must match the dimension used at index
# time in index-knowledge.py.
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8083")
# In-cluster cross-encoder re-ranker. Scores the OpenSearch candidate set and
# keeps only the most relevant chunks, so the auditor prompt stays small.
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8084")
INDEX_NAME = os.environ.get("INDEX_NAME", "k8s-autoscaling-knowledge")
# Candidates pulled from vector search before re-ranking. The cross-encoder
# reranker scores every candidate on CPU, so this count is the main lever on
# rerank latency (~linear in candidates). The tuned retriever already ranks
# well, so 6 candidates keep nearly all the recall of 10 while cutting rerank
# time by ~40%. The orchestrator keeps only the top few of these for the SLM.
RETRIEVE_CANDIDATES = 6
# Query budget (chars) sent to the cross-encoder. The reranker only needs the
# question's intent to score doc relevance, and it re-encodes the query against
# every candidate. Long config pastes blow up rerank latency, so we cap the
# query here (the full text still goes to the auditor SLM untouched).
RERANK_QUERY_CHARS = 512

# --- Read-only tool config ---
#
# The agent may call read-only Kubernetes tools to gather live evidence before
# answering. Set TOOLS_ENABLED=false (or run without cluster credentials) to
# degrade gracefully to RAG-only — the demo still works without tools.
TOOLS_ENABLED = os.environ.get("TOOLS_ENABLED", "true").lower() not in ("0", "false", "no")
# Hard cap on tool calls per query so the plan -> tool -> plan loop can never run
# away (LangGraph's recursion limit is a second backstop).
MAX_TOOL_CALLS = 3

# Kubernetes resource keywords -> API metadata, in priority order (most specific
# first so "nodepool" is matched before "node", "poddisruptionbudget" before
# "pod"). Only read-only kinds the demo reasons about are listed.
RESOURCE_KEYWORDS = [
    ("ec2nodeclass", {"api_version": "karpenter.k8s.aws/v1", "kind": "EC2NodeClass", "namespaced": False}),
    ("nodepool", {"api_version": "karpenter.sh/v1", "kind": "NodePool", "namespaced": False}),
    ("horizontalpodautoscaler", {"api_version": "autoscaling/v2", "kind": "HorizontalPodAutoscaler", "namespaced": True}),
    ("scaledobject", {"api_version": "keda.sh/v1alpha1", "kind": "ScaledObject", "namespaced": True}),
    ("poddisruptionbudget", {"api_version": "policy/v1", "kind": "PodDisruptionBudget", "namespaced": True}),
    ("deployment", {"api_version": "apps/v1", "kind": "Deployment", "namespaced": True}),
    ("hpa", {"api_version": "autoscaling/v2", "kind": "HorizontalPodAutoscaler", "namespaced": True}),
    ("pdb", {"api_version": "policy/v1", "kind": "PodDisruptionBudget", "namespaced": True}),
    ("node", {"api_version": "v1", "kind": "Node", "namespaced": False}),
    ("pod", {"api_version": "v1", "kind": "Pod", "namespaced": True}),
]

# Deprecated apiVersions the client-side validator flags, with the current one.
DEPRECATED_API_VERSIONS = {
    "autoscaling/v2beta1": "autoscaling/v2",
    "autoscaling/v2beta2": "autoscaling/v2",
    "karpenter.sh/v1alpha5": "karpenter.sh/v1",
    "karpenter.sh/v1beta1": "karpenter.sh/v1",
    "karpenter.k8s.aws/v1beta1": "karpenter.k8s.aws/v1",
    "policy/v1beta1": "policy/v1",
    "extensions/v1beta1": "apps/v1 (or networking.k8s.io/v1 for Ingress)",
}

# Signals that the user is describing a runtime problem worth pulling events for.
_EVENT_SIGNALS = (
    "launch", "pending", "stuck", "fail", "event", "scal", "provision",
    "error", "crash", "evict", "terminat", "disrupt", "not work", "not com",
)

# RFC 1123 DNS subdomain: tool args from the (untrusted) extractor must match
# before any are used against the API server.
_K8S_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")

# --- Critic config ---
#
# After a draft answer, a CPU critic scores how well the answer is grounded in
# the retrieved docs + tool evidence. Below the threshold, the agent refines its
# query and retrieves again (up to MAX_CRITIC_RETRIES extra drafts); if it still
# fails, it escalates to the LLM. The Phase C critic is a lexical-overlap
# heuristic standing in for the trained CPU groundedness scorer added later.
# Calibrated on live answers: well-grounded auditor replies score ~0.20-0.25 and
# ungrounded/generic ones near 0, so the default passes normal answers (keeping
# them on CPU) while still catching weak ones. Raise it (env) to demo the loop.
GROUNDEDNESS_THRESHOLD = float(os.environ.get("GROUNDEDNESS_THRESHOLD", "0.15"))
MAX_CRITIC_RETRIES = 1
# Extra candidates/kept-docs pulled when a refine widens the search on retry.
BROADEN_EXTRA = 4

# Common words ignored when measuring answer-vs-context term overlap, so the
# groundedness score reflects substantive (mostly domain) terms.
_STOPWORDS = {
    "the", "and", "for", "you", "your", "with", "this", "that", "are", "can",
    "will", "not", "but", "from", "have", "has", "was", "were", "they", "their",
    "use", "used", "using", "should", "would", "could", "when", "what", "which",
    "into", "out", "via", "per", "set", "see", "any", "all", "may", "also",
    "here", "there", "then", "than", "these", "those", "such", "based", "like",
    "need", "needs", "make", "sure", "want", "does", "doesn", "don", "yes", "no",
}

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

# --- Shared clients (initialized once) ---

bedrock = boto3.client("bedrock-runtime")


def embed_query(text: str) -> list[float]:
    """Embed text via the Slemify-trained retriever (TEI /embed, 768d)."""
    with httpx.Client(timeout=10) as client:
        resp = client.post(f"{EMBEDDING_URL}/embed", json={"inputs": text[:8000]})
        resp.raise_for_status()
        # TEI returns a list of embeddings, one per input.
        return resp.json()[0]


def rerank_docs(query: str, docs: list[str], top_k: int) -> list[str]:
    """Re-rank candidate docs with the cross-encoder, keeping the best top_k.

    Falls back to the original order (truncated) if the reranker is unavailable,
    so retrieval still works even if the reranker pod is down.
    """
    if not docs:
        return []
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{RERANKER_URL}/rerank",
                json={"query": query[:RERANK_QUERY_CHARS], "documents": docs, "top_k": top_k},
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        return [docs[r["index"]] for r in results]
    except Exception as e:
        print(f"  Rerank failed, using vector order: {e}")
        return docs[:top_k]


def _parse_opensearch_url() -> OpenSearch:
    host = OPENSEARCH_URL.replace("http://", "").replace("https://", "")
    hostname, port = host.split(":") if ":" in host else (host, "9200")
    return OpenSearch(
        hosts=[{"host": hostname, "port": int(port)}],
        use_ssl=False,
        verify_certs=False,
    )


opensearch = _parse_opensearch_url()

# --- Read-only Kubernetes tools ---
#
# Every tool talks to the API via the typed/dynamic Python client with
# structured arguments — never a shelled-out kubectl with interpolated user
# text — so there is no shell-injection surface. The extractor's output is
# untrusted, so names/namespaces are validated against the k8s name regex
# before any call. The orchestrator's RBAC grants only get/list/watch.

_k8s_available = False


def _init_k8s():
    """Load cluster credentials once (in-cluster first, then local kubeconfig)."""
    global _k8s_available
    if not TOOLS_ENABLED:
        print("  K8s tools: disabled (TOOLS_ENABLED=false)")
        return
    try:
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
        _k8s_available = True
        print("  K8s tools: ready (read-only)")
    except Exception as e:
        print(f"  K8s tools: unavailable, degrading to RAG-only ({e})")


def _valid_k8s_name(name: str) -> bool:
    return bool(name) and bool(_K8S_NAME_RE.match(name))


def _trim_k8s_object(obj: dict) -> str:
    """Keep the fields that explain a resource's behavior; drop server noise."""
    meta = obj.get("metadata", {}) or {}
    trimmed = {
        "kind": obj.get("kind"),
        "metadata": {"name": meta.get("name"), "namespace": meta.get("namespace")},
        "spec": obj.get("spec"),
    }
    status = obj.get("status") or {}
    if isinstance(status, dict):
        keep = {k: status[k] for k in ("conditions", "phase", "reason", "message", "resources") if k in status}
        if keep:
            trimmed["status"] = keep
    text = yaml.safe_dump(trimmed, default_flow_style=False, sort_keys=False)
    return text[:2000]


def describe_resource(args: dict) -> str:
    """GET a single cluster resource and return a trimmed view (read-only)."""
    if not _k8s_available:
        return "Cluster tools are disabled; answering from documentation only."
    name = args.get("name")
    if not _valid_k8s_name(name or ""):
        return "No valid resource name found in the query."
    ns = args.get("namespace")
    if ns and not _valid_k8s_name(ns):
        return "Invalid namespace."
    try:
        dyn = DynamicClient(k8s_client.ApiClient())
        res = dyn.resources.get(api_version=args["api_version"], kind=args["kind"])
        if args.get("namespaced"):
            obj = res.get(name=name, namespace=ns or "default")
        else:
            obj = res.get(name=name)
        return _trim_k8s_object(obj.to_dict())
    except Exception as e:
        return f"Could not fetch {args.get('kind')}/{name}: {e}"


def list_events(args: dict) -> str:
    """List recent events, optionally for one object (read-only)."""
    if not _k8s_available:
        return "Cluster tools are disabled; answering from documentation only."
    name = args.get("name")
    if name and not _valid_k8s_name(name):
        return "Invalid object name."
    ns = args.get("namespace") or "default"
    if not _valid_k8s_name(ns):
        return "Invalid namespace."
    try:
        v1 = k8s_client.CoreV1Api()
        field_selector = f"involvedObject.name={name}" if name else None
        events = v1.list_namespaced_event(namespace=ns, field_selector=field_selector, limit=25)
        items = events.items or []
        if not items:
            target = name or f"namespace {ns}"
            return f"No recent events for {target}."
        lines = [
            f"{e.last_timestamp or e.event_time} {e.type}/{e.reason}: {(e.message or '').strip()}"
            for e in items[-10:]
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Could not list events: {e}"


def validate_config(args: dict) -> str:
    """Client-side structural + deprecation lint of pasted YAML (no API call).

    Server-side dry-run would require write verbs in RBAC, so the demo keeps
    this purely local: it parses the manifest and flags missing required fields
    and deprecated apiVersions — a deterministic signal the auditor can ground on.
    """
    text = args.get("yaml", "")
    try:
        docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError as e:
        return f"Invalid YAML: {e}"
    findings = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        api_version = doc.get("apiVersion")
        kind = doc.get("kind")
        name = (doc.get("metadata") or {}).get("name")
        label = kind or "resource"
        if not api_version:
            findings.append(f"{label}: missing apiVersion")
        if not kind:
            findings.append("missing kind")
        if not name:
            findings.append(f"{label}: missing metadata.name")
        if api_version in DEPRECATED_API_VERSIONS:
            findings.append(
                f"{label}: apiVersion '{api_version}' is deprecated; use '{DEPRECATED_API_VERSIONS[api_version]}'"
            )
    if not findings:
        return "No structural or deprecation issues found (client-side checks)."
    return "; ".join(findings)


_TOOLS = {
    "describe_resource": describe_resource,
    "list_events": list_events,
    "validate_config": validate_config,
}


def _run_tool(tool: str, args: dict) -> str:
    fn = _TOOLS.get(tool)
    if not fn:
        return f"Unknown tool: {tool}"
    return fn(args)


# --- Heuristic planner + argument extractor (Phase B stubs) ---
#
# These are deliberately simple rules that stand in for the CPU classification
# (tool routing) and extraction models trained in a later phase. They decide
# which read-only tools to call and pull their arguments from the query.

def _looks_like_yaml(text: str) -> bool:
    return bool(re.search(r"^\s*apiVersion:\s*\S", text, re.MULTILINE)) and \
        bool(re.search(r"^\s*kind:\s*\S", text, re.MULTILINE))


def _resource_ref(text: str) -> dict | None:
    low = text.lower()
    for keyword, meta in RESOURCE_KEYWORDS:
        if keyword in low:
            return meta
    return None


def _extract_name(text: str) -> str | None:
    # Prefer an explicitly delimited identifier (backticks or quotes), then a
    # "named/called X" form. Only return it if it's a valid k8s name.
    for pattern in (r"`([^`]+)`", r"\"([^\"]+)\"", r"'([^']+)'",
                    r"\bnamed\s+([a-z0-9][-a-z0-9.]*)", r"\bcalled\s+([a-z0-9][-a-z0-9.]*)"):
        m = re.search(pattern, text)
        if m and _valid_k8s_name(m.group(1)):
            return m.group(1)
    return None


def _extract_namespace(text: str) -> str | None:
    for pattern in (r"-n\s+([a-z0-9][-a-z0-9.]*)", r"--namespace\s+([a-z0-9][-a-z0-9.]*)",
                    r"namespace\s+([a-z0-9][-a-z0-9.]*)", r"\bin\s+the\s+([a-z0-9][-a-z0-9.]*)\s+namespace"):
        m = re.search(pattern, text)
        if m and _valid_k8s_name(m.group(1)):
            return m.group(1)
    return None


def _select_tools(query: str) -> list:
    """Heuristic tool router: pick read-only tools to gather evidence."""
    tools = []
    if _looks_like_yaml(query):
        # A pasted manifest: lint it deterministically (no API, no RBAC needed).
        tools.append("validate_config")
        return tools
    if _k8s_available:
        ref = _resource_ref(query)
        name = _extract_name(query)
        if ref and name:
            tools.append("describe_resource")
            low = query.lower()
            if any(sig in low for sig in _EVENT_SIGNALS):
                tools.append("list_events")
    return tools


def _extract_args(query: str, tool: str) -> dict:
    if tool == "validate_config":
        return {"yaml": query}
    ref = _resource_ref(query) or {}
    return {
        "api_version": ref.get("api_version"),
        "kind": ref.get("kind"),
        "namespaced": ref.get("namespaced", True),
        "name": _extract_name(query),
        "namespace": _extract_namespace(query),
    }


def _args_summary(tool: str, args: dict) -> str:
    if tool == "validate_config":
        return "pasted manifest"
    if tool == "list_events":
        return f"{args.get('name') or '(namespace)'} in {args.get('namespace') or 'default'}"
    return f"{args.get('kind')}/{args.get('name')}"


def _tool_detail(output: str) -> str:
    first = (output or "").strip().split("\n")[0]
    return first[:80]


def _format_tool_results(results: list) -> str:
    parts = []
    for r in results:
        header = f"[tool: {r['tool']} · {_args_summary(r['tool'], r['args'])}]"
        parts.append(f"{header}\n{r['output']}")
    return "\n\n".join(parts)


def _build_context(state: dict) -> str:
    """Assemble the grounding context (tool evidence + retrieved docs)."""
    sections = []
    tool_results = state.get("tool_results", [])
    if tool_results:
        sections.append("LIVE CLUSTER EVIDENCE (read-only tools):\n" + _format_tool_results(tool_results))
    docs = state.get("docs", [])
    if docs:
        sections.append("\n\n---\n\n".join(docs))
    return "\n\n===\n\n".join(sections)


def _content_terms(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.\-/]{2,}", (text or "").lower()) if w not in _STOPWORDS}


def _groundedness(draft: str, context: str) -> float:
    """Fraction of the answer's substantive terms that appear in the context.

    A cheap CPU proxy for groundedness: an answer that reuses the documentation
    and live evidence scores high; a generic or off-topic answer scores low.
    Stands in for the trained groundedness scorer trained in a later phase.
    """
    answer_terms = _content_terms(draft)
    if not answer_terms:
        return 0.0
    context_terms = _content_terms(context)
    if not context_terms:
        return 0.0
    return len(answer_terms & context_terms) / len(answer_terms)

# --- App ---

app = FastAPI()
_ready = False


class Query(BaseModel):
    text: str


def sse(event_type: str, **kwargs) -> str:
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps({'type': event_type, **kwargs})}\n\n"


# --- Health check (gates readiness on warmup) ---

@app.get("/health")
async def health():
    if not _ready:
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "warming up"}, status_code=503)
    return {"status": "ok"}


# --- Startup warmup ---

@app.on_event("startup")
async def warmup():
    """Warm SLMs and Bedrock to avoid cold-start latency on first query."""
    global _ready
    _init_k8s()
    warmup_body = {
        "model": "model",
        "messages": [{"role": "user", "content": (
            "Audit this NodePool:\napiVersion: karpenter.sh/v1\nkind: NodePool\n"
            "metadata:\n  name: test\nspec:\n  template:\n    spec:\n      "
            "requirements:\n        - key: karpenter.k8s.aws/instance-category\n"
            "          operator: In\n          values: [\"c\", \"m\"]"
        )}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            client.post(f"{TRIAGE_URL}/v1/chat/completions", json=warmup_body),
            client.post(f"{AUDITOR_URL}/v1/chat/completions", json=warmup_body),
            return_exceptions=True,
        )
    for name, r in zip(("triage", "auditor"), results):
        status = f"ok ({r.status_code})" if not isinstance(r, Exception) else f"failed ({r})"
        print(f"  Warmup {name}: {status}")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embed_query, "warmup")
        print("  Warmup embedding: ok")
    except Exception as e:
        print(f"  Warmup embedding: failed ({e})")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, rerank_docs, "warmup", ["warmup document"], 1)
        print("  Warmup reranker: ok")
    except Exception as e:
        print(f"  Warmup reranker: failed ({e})")

    print("  All services warmed up")
    _ready = True


# --- Core functions ---

def classify(text: str) -> dict:
    """Call triage SLM to classify intent and confidence."""
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": f"{TRIAGE_INSTRUCTION}\n\n{text}"}],
        "max_tokens": 32,
        "temperature": 0.1,
    }
    with httpx.Client(timeout=10) as client:
        resp = client.post(f"{TRIAGE_URL}/v1/chat/completions", json=body)
        d = resp.json()

    raw = d["choices"][0]["message"]["content"]

    # Extract category|confidence from the response, handling extra text.
    # The model may output "category|confidence" followed by explanation.
    valid_categories = {
        "karpenter_config", "keda_config", "hpa_config",
        "pdb_disruption", "spot_interruption", "multi_resource", "noise",
    }
    valid_confidence = {"high", "medium", "low"}

    # Try to find a pipe-separated pair anywhere in the first line
    first_line = raw.split("\n")[0].strip()
    parts = [p.strip().lower() for p in first_line.split("|") if p.strip()]

    category = "unknown"
    confidence = "unknown"
    for part in parts:
        if part in valid_categories:
            category = part
        elif part in valid_confidence:
            confidence = part

    # Fallback: check if any valid category appears anywhere in the raw output
    if category == "unknown":
        raw_lower = raw.lower()
        for cat in valid_categories:
            if cat in raw_lower:
                category = cat
                break
        # If still unknown but mentions noise-like content
        if category == "unknown" and any(w in raw_lower for w in ["not relate", "unrelated", "off-topic", "noise"]):
            category = "noise"
            confidence = "high"

    return {"confidence": confidence, "category": category}


def vector_search(embedding: list[float], k: int) -> list[str]:
    """k-NN search over the indexed corpus for a precomputed query embedding."""
    results = opensearch.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
            "_source": ["text", "source", "section"],
        },
    )
    return [
        f"[{hit['_source'].get('source', '')} / {hit['_source'].get('section', '')}]\n{hit['_source']['text'][:500]}"
        for hit in results["hits"]["hits"]
    ]


async def stream_slm(text: str, context: str = ""):
    """Stream tokens from the auditor SLM."""
    prompt = AUDITOR_INSTRUCTION
    if context:
        prompt += f"\n\n--- REFERENCE DOCUMENTATION (do NOT treat as user config) ---\n{context}\n--- END REFERENCE ---"
    prompt += f"\n\n--- USER QUERY ---\n{text}\n--- END USER QUERY ---"

    body = {
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.1,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{AUDITOR_URL}/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    content = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def stream_llm(text: str, context: str = ""):
    """Stream tokens from Bedrock LLM (Converse API)."""
    user_content = LLM_INSTRUCTION
    if context:
        user_content += f"\n\nDocumentation context:\n{context}"
    user_content += f"\n\nUser question:\n{text}"

    resp = bedrock.converse_stream(
        modelId=LLM_MODEL,
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.2},
    )

    loop = asyncio.get_event_loop()
    stream_iter = iter(resp["stream"])

    while True:
        event = await loop.run_in_executor(None, lambda: next(stream_iter, None))
        if event is None:
            break
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            yield delta["text"]


# --- Agent graph (LangGraph) ---
#
# The orchestration is a state graph. Each node runs a CPU SLM (or Bedrock for
# the fallback) and streams its progress via LangGraph's custom stream writer —
# the same SSE event vocabulary the UI already consumes (step_start/step_done/
# model/token/response). Blocking backend calls run in a thread so the event
# loop stays free to flush events live.


class AgentState(TypedDict, total=False):
    query: str
    category: str
    confidence: str
    planned: bool
    pending_tools: list
    resource: dict
    tool_results: list
    embedding: list
    candidates: list
    docs: list
    effective_query: str
    broaden: bool
    draft: str
    attempts: int
    critic_score: float
    used_llm: bool


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


async def n_triage(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    writer({"type": "step_start", "name": "Triage SLM (4B, CPU)", "note": "classifying intent"})
    t = time.perf_counter()
    result = await loop.run_in_executor(None, classify, state["query"])
    writer({"type": "step_done", "name": "Triage SLM (4B, CPU)", "ms": _ms(t),
            "detail": f"{result['category'].replace('_', ' ')} · {result['confidence']} confidence"})
    return {"category": result["category"], "confidence": result["confidence"]}


async def n_reject(state: AgentState) -> dict:
    get_stream_writer()({"type": "response",
                         "text": "This does not look like a K8s autoscaling question."})
    return {}


async def n_plan(state: AgentState) -> dict:
    """Decide which read-only tools (if any) to call before answering.

    First pass selects the tool list and emits a planner step; later passes
    (re-entered after each tool) are silent decisions and enforce the call cap.
    """
    if state.get("planned"):
        if len(state.get("tool_results", [])) >= MAX_TOOL_CALLS:
            return {"pending_tools": []}
        return {}
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    writer({"type": "step_start", "name": "Planner (CPU)", "note": "deciding tool use"})
    t = time.perf_counter()
    tools = await loop.run_in_executor(None, _select_tools, state["query"])
    detail = ("tools: " + ", ".join(tools)) if tools else "no tools needed — answering from docs"
    writer({"type": "step_done", "name": "Planner (CPU)", "ms": _ms(t), "detail": detail})
    return {"planned": True, "pending_tools": tools}


async def n_extract_args(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    tool = state["pending_tools"][0]
    writer({"type": "step_start", "name": "Arg extractor (CPU)", "note": f"args for {tool}"})
    t = time.perf_counter()
    args = await loop.run_in_executor(None, _extract_args, state["query"], tool)
    writer({"type": "step_done", "name": "Arg extractor (CPU)", "ms": _ms(t),
            "detail": _args_summary(tool, args)})
    return {"resource": args}


async def n_run_tool(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    pending = state["pending_tools"]
    tool, rest = pending[0], pending[1:]
    args = state.get("resource", {})
    writer({"type": "step_start", "name": f"Tool · {tool}", "note": _args_summary(tool, args)})
    t = time.perf_counter()
    output = await loop.run_in_executor(None, _run_tool, tool, args)
    writer({"type": "step_done", "name": f"Tool · {tool}", "ms": _ms(t), "detail": _tool_detail(output)})
    results = state.get("tool_results", []) + [{"tool": tool, "args": args, "output": output}]
    return {"pending_tools": rest, "tool_results": results}


async def n_embed(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    query = state.get("effective_query") or state["query"]
    writer({"type": "step_start", "name": "Retriever (tuned encoder, CPU)", "note": "embedding query → 768d"})
    t = time.perf_counter()
    embedding = await loop.run_in_executor(None, embed_query, query)
    writer({"type": "step_done", "name": "Retriever (tuned encoder, CPU)", "ms": _ms(t),
            "detail": "domain-tuned ONNX encoder"})
    return {"embedding": embedding}


async def n_search(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    k = RETRIEVE_CANDIDATES + (BROADEN_EXTRA if state.get("broaden") else 0)
    writer({"type": "step_start", "name": "OpenSearch (vector DB)", "note": f"k-NN search, top {k}"})
    t = time.perf_counter()
    candidates = await loop.run_in_executor(None, vector_search, state["embedding"], k)
    writer({"type": "step_done", "name": "OpenSearch (vector DB)", "ms": _ms(t),
            "detail": f"{len(candidates)} candidate chunks"})
    return {"candidates": candidates}


async def n_rerank(state: AgentState) -> dict:
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    keep_k = 5 if state["confidence"] in ("low", "unknown") else 2
    if state.get("broaden"):
        keep_k += 2
    candidates = state["candidates"]
    query = state.get("effective_query") or state["query"]
    writer({"type": "step_start", "name": "Reranker (cross-encoder, CPU)",
            "note": f"scoring {len(candidates)} pairs → top {keep_k}"})
    t = time.perf_counter()
    docs = await loop.run_in_executor(None, rerank_docs, query, candidates, keep_k)
    writer({"type": "step_done", "name": "Reranker (cross-encoder, CPU)", "ms": _ms(t),
            "detail": f"kept top {len(docs)}"})
    return {"docs": docs}


async def n_generate(state: AgentState) -> dict:
    writer = get_stream_writer()
    low_conf = state["confidence"] in ("low", "unknown")
    context = _build_context(state)
    attempts = state.get("attempts", 0)
    if attempts > 0:
        # This is a retry after the critic rejected the previous draft; tell the
        # UI to start a fresh answer block so drafts don't concatenate.
        writer({"type": "answer_reset", "reason": "refining"})

    if low_conf:
        gen_name, stream_fn, used_llm = "LLM API (Bedrock fallback)", stream_llm, True
        writer({"type": "model", "name": "Claude Sonnet 4.5 (Bedrock)"})
    else:
        gen_name, stream_fn, used_llm = "Auditor SLM (8B, CPU)", stream_slm, False
        writer({"type": "model", "name": "Auditor SLM (8B, CPU)"})

    writer({"type": "step_start", "name": gen_name, "note": "generating answer"})
    t = time.perf_counter()
    first = True
    parts = []
    async for token in stream_fn(state["query"], context):
        if first:
            writer({"type": "step_done", "name": gen_name, "ms": _ms(t), "detail": "time to first token"})
            first = False
        parts.append(token)
        writer({"type": "token", "text": token})
    if first:
        writer({"type": "step_done", "name": gen_name, "ms": _ms(t), "detail": "no output"})
    return {"draft": "".join(parts), "attempts": attempts + 1, "used_llm": used_llm}


async def n_critic(state: AgentState) -> dict:
    """Score how well the draft is grounded in the retrieved evidence (CPU)."""
    if state.get("used_llm"):
        # The LLM is the top of the escalation ladder — nothing to escalate to,
        # so don't re-critique an LLM answer.
        return {}
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    context = _build_context(state)
    writer({"type": "step_start", "name": "Critic (CPU)", "note": "scoring groundedness"})
    t = time.perf_counter()
    score = await loop.run_in_executor(None, _groundedness, state.get("draft", ""), context)
    attempts = state.get("attempts", 0)
    if score >= GROUNDEDNESS_THRESHOLD:
        verdict = "pass"
    elif attempts <= MAX_CRITIC_RETRIES:
        verdict = "refining"
    else:
        verdict = "escalating"
    writer({"type": "step_done", "name": "Critic (CPU)", "ms": _ms(t),
            "detail": f"groundedness {score:.2f} · {verdict}"})
    return {"critic_score": score}


async def n_refine(state: AgentState) -> dict:
    """Broaden the retrieval query after a weakly-grounded draft."""
    writer = get_stream_writer()
    writer({"type": "step_start", "name": "Refine (CPU)", "note": "broadening retrieval"})
    t = time.perf_counter()
    category = state.get("category", "").replace("_", " ")
    expanded = f"{state['query']} {category} configuration best practices troubleshooting".strip()
    writer({"type": "step_done", "name": "Refine (CPU)", "ms": _ms(t),
            "detail": "expanded query, widening search"})
    return {"effective_query": expanded, "broaden": True}


async def n_escalate(state: AgentState) -> dict:
    """The CPU critic kept failing — escalate to the LLM with gathered context."""
    writer = get_stream_writer()
    context = _build_context(state)
    writer({"type": "answer_reset", "reason": "escalating"})
    writer({"type": "model", "name": "Claude Sonnet 4.5 (Bedrock)"})
    writer({"type": "step_start", "name": "LLM API (Bedrock escalation)",
            "note": "CPU critic kept failing — escalating"})
    t = time.perf_counter()
    first = True
    async for token in stream_llm(state["query"], context):
        if first:
            writer({"type": "step_done", "name": "LLM API (Bedrock escalation)", "ms": _ms(t),
                    "detail": "time to first token"})
            first = False
        writer({"type": "token", "text": token})
    if first:
        writer({"type": "step_done", "name": "LLM API (Bedrock escalation)", "ms": _ms(t), "detail": "no output"})
    return {"used_llm": True}


def _route_after_triage(state: AgentState) -> str:
    return "reject" if state["category"] == "noise" else "plan"


def _route_after_plan(state: AgentState) -> str:
    """Loop into the tool path while tools remain; otherwise retrieve."""
    return "extract_args" if state.get("pending_tools") else "embed"


def _route_after_critic(state: AgentState) -> str:
    """Pass to the user, loop back to refine retrieval, or escalate to the LLM."""
    if state.get("used_llm"):
        return "end"
    if state.get("critic_score", 0.0) >= GROUNDEDNESS_THRESHOLD:
        return "end"
    if state.get("attempts", 0) <= MAX_CRITIC_RETRIES:
        return "refine"
    return "escalate"


def _build_agent():
    g = StateGraph(AgentState)
    g.add_node("triage", n_triage)
    g.add_node("reject", n_reject)
    g.add_node("plan", n_plan)
    g.add_node("extract_args", n_extract_args)
    g.add_node("run_tool", n_run_tool)
    g.add_node("embed", n_embed)
    g.add_node("search", n_search)
    g.add_node("rerank", n_rerank)
    g.add_node("generate", n_generate)
    g.add_node("critic", n_critic)
    g.add_node("refine", n_refine)
    g.add_node("escalate", n_escalate)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", _route_after_triage, {"reject": "reject", "plan": "plan"})
    g.add_edge("reject", END)
    g.add_conditional_edges("plan", _route_after_plan, {"extract_args": "extract_args", "embed": "embed"})
    g.add_edge("extract_args", "run_tool")
    g.add_edge("run_tool", "plan")
    g.add_edge("embed", "search")
    g.add_edge("search", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "critic")
    g.add_conditional_edges("critic", _route_after_critic,
                            {"end": END, "refine": "refine", "escalate": "escalate"})
    g.add_edge("refine", "embed")
    g.add_edge("escalate", END)
    return g.compile()


agent = _build_agent()


# --- Route handler ---

@app.post("/query")
async def query_endpoint(q: Query):
    async def event_stream():
        t_start = time.perf_counter()
        # stream_mode="custom" yields exactly the dicts each node writes, so the
        # UI's SSE contract is preserved without LangChain message plumbing.
        async for event in agent.astream({"query": q.text}, stream_mode="custom"):
            yield f"data: {json.dumps(event)}\n\n"
        yield sse("total", ms=round((time.perf_counter() - t_start) * 1000))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- UI ---

@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = """<!DOCTYPE html>
<html data-theme="dark"><head><meta charset="UTF-8"><title>K8s Autoscaling Expert</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root[data-theme="dark"]{--bg:#0f1117;--fg:#e6edf3;--surface:#161b22;--border:#30363d;--muted:#8b949e;--accent:#1f6feb;--accent-hover:#388bfd;--code-bg:#0d1117;--strong:#f0f6fc;--user-bg:#1f6feb;--slm-color:#3fb950;--slm-border:#238636;--slm-bg:rgba(63,185,80,0.1);--llm-color:#a371f7;--llm-border:#8957e5;--llm-bg:rgba(163,113,247,0.1)}
:root[data-theme="light"]{--bg:#ffffff;--fg:#1f2328;--surface:#f6f8fa;--border:#d1d9e0;--muted:#656d76;--accent:#0969da;--accent-hover:#0550ae;--code-bg:#f6f8fa;--strong:#1f2328;--user-bg:#0969da;--slm-color:#1a7f37;--slm-border:#1a7f37;--slm-bg:rgba(26,127,55,0.08);--llm-color:#8250df;--llm-border:#8250df;--llm-bg:rgba(130,80,223,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column;transition:background .2s,color .2s}
.header{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:18px;font-weight:600}
.header .actions{display:flex;gap:8px}
.header button{background:transparent;border:1px solid var(--border);color:var(--muted);padding:6px 12px;border-radius:6px;cursor:pointer;font-size:12px}
.header button:hover{border-color:var(--accent);color:var(--accent)}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:85%;padding:12px 16px;border-radius:12px;font-size:14px;line-height:1.7}
.msg.user{align-self:flex-end;background:var(--user-bg);color:white;white-space:pre-wrap}
.msg.system{align-self:flex-start;background:var(--surface);border:1px solid var(--border)}
.msg.status{align-self:flex-start;color:var(--muted);font-size:12px;padding:4px 0;display:flex;align-items:center;gap:6px}
.msg.status::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--muted);animation:pulse 1.5s infinite}
.msg.step{align-self:stretch;max-width:100%;display:flex;align-items:center;gap:10px;background:var(--surface);border:1px solid var(--border);padding:8px 12px;border-radius:8px;font-size:13px}
.msg.step .step-name{font-weight:600}
.msg.step .step-note{color:var(--muted);font-size:12px;flex:1}
.msg.step .dur{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:600;color:var(--accent)}
.msg.step.done .dur{color:var(--slm-color)}
.msg.step .spinner{width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;display:inline-block;flex-shrink:0}
.msg.step .check{color:var(--slm-color);font-weight:700;flex-shrink:0}
.msg.total{align-self:stretch;max-width:100%;text-align:right;color:var(--strong);font-weight:600;font-size:13px;padding:4px 12px;border-top:1px solid var(--border)}
.msg.model-badge{align-self:flex-start;font-size:11px;padding:4px 10px;border-radius:20px;font-weight:500;border:1px solid}
.msg.model-badge.slm{color:var(--slm-color);border-color:var(--slm-border);background:var(--slm-bg)}
.msg.model-badge.llm{color:var(--llm-color);border-color:var(--llm-border);background:var(--llm-bg)}
.msg h1,.msg h2,.msg h3{margin:12px 0 6px;font-weight:600}
.msg h1{font-size:16px}.msg h2{font-size:15px}.msg h3{font-size:14px}
.msg p{margin:6px 0}
.msg ul,.msg ol{margin:6px 0;padding-left:20px}
.msg li{margin:3px 0}
.msg pre{background:var(--code-bg);padding:12px;border-radius:6px;overflow-x:auto;margin:8px 0;font-size:13px;border:1px solid var(--border)}
.msg code{font-size:13px;font-family:'SF Mono',Menlo,monospace;background:var(--code-bg);padding:2px 5px;border-radius:3px}
.msg pre code{background:none;padding:0}
.msg strong{color:var(--strong)}
.msg hr{border:none;border-top:1px solid var(--border);margin:12px 0}
.input-area{padding:16px 24px;border-top:1px solid var(--border);display:flex;gap:12px}
.input-area textarea{flex:1;background:var(--surface);border:1px solid var(--border);color:var(--fg);padding:12px;border-radius:8px;font-size:14px;resize:none;height:80px;font-family:inherit}
.input-area textarea:focus{outline:none;border-color:var(--accent)}
.input-area button{background:var(--accent);color:white;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:500}
.input-area button:hover{background:var(--accent-hover)}
.input-area button:disabled{opacity:0.5;cursor:not-allowed}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<div class="header">
  <h1>K8s Autoscaling Expert</h1>
  <div class="actions">
    <button onclick="toggleTheme()">Light/Dark</button>
    <button onclick="clearChat()">Clear</button>
  </div>
</div>
<div class="chat" id="chat"></div>
<div class="input-area">
  <textarea id="input" placeholder="Paste a K8s config or ask a question..."></textarea>
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const btn = document.getElementById('send');

function toggleTheme() {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
}

function addMsg(html, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function renderMd(text) {
  try { return marked.parse(text); } catch(e) { return text; }
}

let steps = {};
function fmt(ms) { return ms >= 1000 ? (ms/1000).toFixed(2) + 's' : ms + ' ms'; }

function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function addStep(name, note) {
  const div = document.createElement('div');
  div.className = 'msg step running';
  div.innerHTML = '<span class="spinner"></span>' +
    '<span class="step-name">' + escapeHtml(name) + '</span>' +
    '<span class="step-note">' + escapeHtml(note || '') + '</span>' +
    '<span class="dur"></span>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  steps[name] = div;
}

function finishStep(name, ms, detail) {
  const div = steps[name];
  if (!div) return;
  div.className = 'msg step done';
  const spinner = div.querySelector('.spinner');
  if (spinner) { const c = document.createElement('span'); c.className = 'check'; c.textContent = '\u2713'; spinner.replaceWith(c); }
  if (detail) div.querySelector('.step-note').textContent = detail;
  div.querySelector('.dur').textContent = fmt(ms);
}

function clearChat() { chat.innerHTML = ''; }

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  btn.disabled = true;
  steps = {};
  addMsg(text.replace(/</g,'&lt;').replace(/\\n/g,'<br>'), 'user');

  const resp = await fetch('/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let responseDiv = null, rawText = '', buffer = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split('\\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') {
        if (responseDiv && rawText) responseDiv.innerHTML = renderMd(rawText);
        break;
      }
      try {
        const msg = JSON.parse(data);
        if (msg.type === 'step_start') {
          addStep(msg.name, msg.note);
        } else if (msg.type === 'step_done') {
          finishStep(msg.name, msg.ms, msg.detail);
        } else if (msg.type === 'total') {
          addMsg('Total pipeline time: <strong>' + fmt(msg.ms) + '</strong>', 'total');
        } else if (msg.type === 'model') {
          const isSlm = msg.name.toLowerCase().includes('slm');
          addMsg(msg.name, 'model-badge ' + (isSlm ? 'slm' : 'llm'));
        } else if (msg.type === 'response') {
          addMsg(renderMd(msg.text), 'system');
        } else if (msg.type === 'answer_reset') {
          if (responseDiv && rawText) responseDiv.innerHTML = renderMd(rawText);
          responseDiv = null; rawText = '';
          const note = msg.reason === 'escalating'
            ? 'CPU critic kept failing \u2014 escalating to the LLM.'
            : 'Draft was weakly grounded \u2014 refining and retrying.';
          addMsg(note, 'status');
        } else if (msg.type === 'token') {
          if (!responseDiv) responseDiv = addMsg('', 'system');
          rawText += msg.text;
          responseDiv.innerHTML = renderMd(rawText);
          chat.scrollTop = chat.scrollHeight;
        }
      } catch(e) {}
    }
  }
  btn.disabled = false;
  input.focus();
}

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }});
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
