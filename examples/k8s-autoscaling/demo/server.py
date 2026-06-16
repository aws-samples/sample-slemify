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


# Words that signal an operational / live-cluster question (a runtime symptom or
# a cost/provisioning concern). Used only to ROUTE to a live investigation; the
# investigation itself is state-driven, not keyed to these words.
_OPERATIONAL_SIGNALS = _EVENT_SIGNALS + (
    "cost", "expensive", "bill", "spot", "saving", "price", "cheaper",
    "on-demand", "ondemand",
)


def _validate_draft_fix(draft: str) -> list:
    """Lint any YAML the draft proposes, against current-API knowledge.

    Reuses validate_config (which knows the deprecated apiVersions from the
    docs), so the agent can catch a fix that uses stale or invalid configuration
    before showing or applying it — grounding the *fix*, not just the diagnosis,
    in current standards.
    """
    issues = []
    for block in re.findall(r"```(?:ya?ml)?\s*\n(.*?)```", draft or "", re.DOTALL):
        if "apiVersion" in block and "kind" in block:
            result = validate_config({"yaml": block})
            if result and not result.startswith("No structural"):
                issues.append(result)
    return issues


def _pod_scheduling_summary(pod) -> str:
    """The scheduling-relevant fields of a pod, focused for the auditor."""
    spec = pod.spec
    sel = dict(spec.node_selector) if spec.node_selector else None
    has_affinity = bool(spec.affinity and spec.affinity.node_affinity)
    tols = [t.key for t in (spec.tolerations or []) if getattr(t, "key", None)]
    return (f"    nodeSelector: {sel or 'none'}; "
            f"nodeAffinity: {'set' if has_affinity else 'none'}; "
            f"tolerations: {tols or 'none'}")


def _latest_warning_event(v1, namespace: str, name: str) -> str | None:
    try:
        evs = v1.list_namespaced_event(
            namespace=namespace, field_selector=f"involvedObject.name={name}", limit=25).items or []
        warns = [e for e in evs if e.type == "Warning"]
        if not warns:
            return None
        e = warns[-1]
        return f"{e.reason}: {(e.message or '').strip()[:280]}"
    except Exception:
        return None


def _nodepool_summaries() -> str:
    """Focused per-NodePool view of the cost/provisioning-relevant requirements."""
    try:
        dyn = DynamicClient(k8s_client.ApiClient())
        res = dyn.resources.get(api_version="karpenter.sh/v1", kind="NodePool")
        items = res.get().items
    except Exception as e:
        return f"(could not list NodePools: {e})"
    out = []
    for np in items:
        d = np.to_dict()
        name = (d.get("metadata") or {}).get("name")
        reqs = (((d.get("spec") or {}).get("template") or {}).get("spec") or {}).get("requirements") or []
        fields = {}
        for r in reqs:
            if isinstance(r, dict):
                fields[r.get("key")] = r.get("values")
        disruption = ((d.get("spec") or {}).get("disruption") or {}).get("consolidationPolicy")
        out.append(
            f"- {name}: capacity-type={fields.get('karpenter.sh/capacity-type') or 'any'}, "
            f"instance-family={fields.get('karpenter.k8s.aws/instance-family') or '-'}, "
            f"instance-category={fields.get('karpenter.k8s.aws/instance-category') or '-'}, "
            f"arch={fields.get('kubernetes.io/arch') or 'any'}, consolidation={disruption or '-'}"
        )
    return "\n".join(out) if out else "(no NodePools found)"


def investigate_cluster(args: dict) -> str:
    """Read-only first-look triage: survey unhealthy pods (+ their events) and
    NodePool provisioning config, returning a focused evidence bundle.

    This is general SRE triage (the `kubectl get pods` / `describe` / `get events`
    a human runs first), not problem-specific logic. The auditor reasons over the
    gathered evidence to find the root cause.
    """
    if not _k8s_available:
        return "Cluster tools are disabled; answering from documentation only."
    namespace = args.get("namespace")
    if namespace and not _valid_k8s_name(namespace):
        return "Invalid namespace."
    query = (args.get("query") or "").lower()

    # State-driven triage (general, not keyed to query wording): if any pods are
    # unhealthy, diagnose those; otherwise the issue is more likely the NodePool
    # provisioning config, so report that.
    v1 = k8s_client.CoreV1Api()

    # --- Pod-health axis: which pods are unhealthy and why ---
    try:
        pods = (v1.list_namespaced_pod(namespace).items if namespace
                else v1.list_pod_for_all_namespaces().items)
    except Exception as e:
        return f"(could not list pods: {e})\n\nNODEPOOLS:\n" + _nodepool_summaries()
    problems = []
    for p in pods:
        phase = p.status.phase
        cs = p.status.container_statuses or []
        ready = bool(cs) and all(c.ready for c in cs)
        if phase == "Pending" or (phase == "Running" and not ready):
            problems.append(p)
    if problems:
        scope = f"namespace {namespace}" if namespace else "the cluster"
        lines = [f"{len(problems)} pod(s) not healthy in {scope}:"]
        for p in problems[:5]:
            pn, pns = p.metadata.name, p.metadata.namespace
            lines.append(f"  {pns}/{pn} [{p.status.phase}]")
            lines.append(_pod_scheduling_summary(p))
            event = _latest_warning_event(v1, pns, pn)
            if event:
                lines.append(f"    event: {event}")
        return "PROBLEM PODS:\n" + "\n".join(lines)

    # No unhealthy pods found — the issue is more likely provisioning config.
    return "NODEPOOLS:\n" + _nodepool_summaries()


_TOOLS = {
    "describe_resource": describe_resource,
    "list_events": list_events,
    "validate_config": validate_config,
    "investigate_cluster": investigate_cluster,
}


def _run_tool(tool: str, args: dict) -> str:
    fn = _TOOLS.get(tool)
    if not fn:
        return f"Unknown tool: {tool}"
    return fn(args)


# --- Remediation (write actions, gated and bounded) ---
#
# Apply is OFF unless ALLOW_APPLY is set AND the orchestrator has write RBAC.
# Remediations are a small whitelist of deterministic, validated patches against
# a SPECIFIC named target — never free-form model YAML, never a blanket change.
# Each dry-runs (server-side) before applying, touches only the intended field,
# and is followed by a re-read to verify.
ALLOW_APPLY = os.environ.get("ALLOW_APPLY", "false").lower() in ("1", "true", "yes")


def _nodepool_capacity_types(np: dict):
    reqs = (((np.get("spec") or {}).get("template") or {}).get("spec") or {}).get("requirements") or []
    for r in reqs:
        if r.get("key") == "karpenter.sh/capacity-type":
            return r.get("values")
    return None


def enable_spot_on_nodepool(name: str) -> dict:
    """Add 'spot' to one named NodePool's capacity-type. Deterministic patch:
    dry-runs first, changes only the capacity-type values, nothing else."""
    if not ALLOW_APPLY:
        return {"ok": False, "message": "Apply is disabled (set ALLOW_APPLY=true and grant write RBAC)."}
    if not _valid_k8s_name(name):
        return {"ok": False, "message": "Invalid NodePool name."}
    api = k8s_client.CustomObjectsApi()
    try:
        np = api.get_cluster_custom_object("karpenter.sh", "v1", "nodepools", name)
    except Exception as e:
        return {"ok": False, "message": f"Could not fetch NodePool {name}: {e}"}
    reqs = (((np.get("spec") or {}).get("template") or {}).get("spec") or {}).get("requirements") or []
    amended = False
    for r in reqs:
        if r.get("key") == "karpenter.sh/capacity-type":
            vals = r.get("values") or []
            if "spot" in vals:
                return {"ok": True, "message": f"NodePool {name} already allows Spot; no change needed."}
            r["values"] = sorted(set(vals) | {"spot"})
            amended = True
    if not amended:
        return {"ok": False, "message": f"NodePool {name} has no capacity-type requirement to amend."}
    patch = {"spec": {"template": {"spec": {"requirements": reqs}}}}
    try:
        api.patch_cluster_custom_object("karpenter.sh", "v1", "nodepools", name, body=patch, dry_run="All")
        api.patch_cluster_custom_object("karpenter.sh", "v1", "nodepools", name, body=patch)
    except Exception as e:
        return {"ok": False, "message": f"Apply failed (dry-run or apply): {e}"}
    return {"ok": True, "message": f"Patched NodePool {name}: capacity-type now includes 'spot'."}


def verify_nodepool_spot(name: str) -> dict:
    """Re-read the NodePool to confirm the remediation took effect."""
    api = k8s_client.CustomObjectsApi()
    try:
        np = api.get_cluster_custom_object("karpenter.sh", "v1", "nodepools", name)
    except Exception as e:
        return {"ok": False, "message": f"Could not re-read NodePool {name}: {e}"}
    caps = _nodepool_capacity_types(np)
    if caps and "spot" in caps:
        return {"ok": True, "message": f"Verified: NodePool {name} capacity-type is now {caps}."}
    return {"ok": False, "message": f"Verification failed: capacity-type is {caps}."}


# Whitelist of structured remediations the agent may apply. Each entry maps to an
# (apply, verify) pair operating on a single named target.
_REMEDIATIONS = {
    "enable_spot_on_nodepool": (enable_spot_on_nodepool, verify_nodepool_spot),
}


def detect_remediation(query: str) -> dict | None:
    """Map a query to a safe, applicable remediation on an EXPLICITLY NAMED
    target. Returns None unless the user named a NodePool that genuinely lacks
    Spot — so apply is always bounded to a resource the user pointed at, never a
    blanket change across the cluster."""
    if not ALLOW_APPLY:
        return None
    ref = _resource_ref(query)
    name = _extract_name(query)
    if not (ref and ref.get("kind") == "NodePool" and name):
        return None
    try:
        np = k8s_client.CustomObjectsApi().get_cluster_custom_object(
            "karpenter.sh", "v1", "nodepools", name)
    except Exception:
        return None
    caps = _nodepool_capacity_types(np)
    if caps is not None and "spot" not in caps:
        return {"action": "enable_spot_on_nodepool", "target": name,
                "summary": f"add 'spot' to NodePool {name} capacity-type"}
    return None


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
    for pattern in (r"-n\s+`?([a-z0-9][-a-z0-9.]*)", r"--namespace\s+`?([a-z0-9][-a-z0-9.]*)",
                    r"namespace\s+[`'\"]?([a-z0-9][-a-z0-9.]*)", r"\bin\s+the\s+[`'\"]?([a-z0-9][-a-z0-9.]*)[`'\"]?\s+namespace"):
        m = re.search(pattern, text)
        if m and _valid_k8s_name(m.group(1)):
            return m.group(1)
    return None


def _select_tools(query: str) -> list:
    """Heuristic tool router: pick read-only tools to gather evidence.

    A pasted manifest is linted; an operational/cost question triggers a live
    cluster investigation; a specific named resource is described directly.
    """
    if _looks_like_yaml(query):
        # A pasted manifest: lint it deterministically (no API, no RBAC needed).
        return ["validate_config"]
    if not _k8s_available:
        return []
    low = query.lower()
    if any(sig in low for sig in _OPERATIONAL_SIGNALS):
        # An operational symptom or cost/provisioning concern: investigate live state.
        return ["investigate_cluster"]
    ref = _resource_ref(query)
    name = _extract_name(query)
    if ref and name:
        # A specific resource named without a symptom: inspect it directly.
        return ["describe_resource"]
    return []


def _extract_args(query: str, tool: str) -> dict:
    if tool == "validate_config":
        return {"yaml": query}
    if tool == "investigate_cluster":
        return {"namespace": _extract_namespace(query), "query": query}
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
    if tool == "investigate_cluster":
        return f"namespace {args.get('namespace') or 'all'}"
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
    autopilot: bool = False


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
    critic_pass: bool
    correction: str
    used_llm: bool
    autopilot: bool


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
    correction = state.get("correction")
    if correction:
        # The critic rejected the previous fix (e.g. deprecated API); steer the
        # regeneration to correct it, grounded in current standards.
        context += "\n\n=== CORRECTION REQUIRED ===\n" + correction
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
    """Check the draft: groundedness (CPU) + validate the proposed fix vs the KB.

    Runs for both SLM and LLM answers. Groundedness only gates the SLM path (the
    LLM is the top of the escalation ladder); the fix-validation gates both, so a
    deprecated/invalid fix is caught and corrected regardless of which model
    wrote it.
    """
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    draft = state.get("draft", "")
    context = _build_context(state)
    used_llm = state.get("used_llm", False)
    attempts = state.get("attempts", 0)
    writer({"type": "step_start", "name": "Critic (CPU)", "note": "scoring groundedness + validating fix"})
    t = time.perf_counter()
    score = await loop.run_in_executor(None, _groundedness, draft, context)
    fix_issues = await loop.run_in_executor(None, _validate_draft_fix, draft)

    grounded_ok = used_llm or score >= GROUNDEDNESS_THRESHOLD
    fix_ok = not fix_issues
    passed = grounded_ok and fix_ok
    can_retry = attempts <= MAX_CRITIC_RETRIES
    if passed:
        verdict = "pass"
    elif can_retry:
        verdict = "refining"
    else:
        verdict = "accepting" if used_llm else "escalating"

    detail = f"groundedness {score:.2f}"
    if fix_issues:
        detail += " · fix uses deprecated/invalid config"
    detail += f" · {verdict}"
    writer({"type": "step_done", "name": "Critic (CPU)", "ms": _ms(t), "detail": detail})

    correction = ""
    if fix_issues and verdict == "refining":
        correction = ("Your previous draft proposed deprecated or invalid configuration: "
                      + "; ".join(fix_issues)
                      + ". Re-issue the fix using only current, non-deprecated APIs from the documentation.")
    return {"critic_score": score, "critic_pass": passed, "correction": correction}


async def n_refine(state: AgentState) -> dict:
    """Broaden retrieval (and/or carry a fix correction) before regenerating."""
    writer = get_stream_writer()
    correcting = bool(state.get("correction"))
    note = "correcting the proposed fix" if correcting else "broadening retrieval"
    writer({"type": "step_start", "name": "Refine (CPU)", "note": note})
    t = time.perf_counter()
    category = state.get("category", "").replace("_", " ")
    expanded = f"{state['query']} {category} configuration best practices troubleshooting".strip()
    detail = "re-grounding the fix in current docs" if correcting else "expanded query, widening search"
    writer({"type": "step_done", "name": "Refine (CPU)", "ms": _ms(t), "detail": detail})
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


async def n_remediate(state: AgentState) -> dict:
    """Optionally apply a structured fix (autopilot) and verify it took effect.

    Read-only by default: only acts when autopilot is on AND apply is enabled
    AND a bounded remediation is detected for a NodePool the user named. The
    apply itself dry-runs, patches one field, and is followed by a verify re-read.
    """
    if not (ALLOW_APPLY and state.get("autopilot")):
        return {}
    writer = get_stream_writer()
    loop = asyncio.get_event_loop()
    rem = await loop.run_in_executor(None, detect_remediation, state["query"])
    if not rem:
        return {}
    action, target = rem["action"], rem["target"]
    apply_fn, verify_fn = _REMEDIATIONS[action]
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


def _route_after_triage(state: AgentState) -> str:
    return "reject" if state["category"] == "noise" else "plan"


def _route_after_plan(state: AgentState) -> str:
    """Loop into the tool path while tools remain; otherwise retrieve."""
    return "extract_args" if state.get("pending_tools") else "embed"


def _route_after_critic(state: AgentState) -> str:
    """Pass to the user, loop back to refine (regenerate with a correction), or
    escalate to the LLM when the CPU path is exhausted."""
    if state.get("critic_pass"):
        return "end"
    if state.get("attempts", 0) <= MAX_CRITIC_RETRIES:
        return "refine"
    # Retries exhausted: an SLM draft escalates to the LLM; an LLM draft has
    # nowhere higher to go, so accept it.
    return "end" if state.get("used_llm") else "escalate"


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
    g.add_node("remediate", n_remediate)
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
                            {"end": "remediate", "refine": "refine", "escalate": "escalate"})
    g.add_edge("refine", "embed")
    g.add_edge("escalate", END)
    g.add_edge("remediate", END)
    return g.compile()


agent = _build_agent()


# --- Route handler ---

@app.post("/query")
async def query_endpoint(q: Query):
    async def event_stream():
        t_start = time.perf_counter()
        # stream_mode="custom" yields exactly the dicts each node writes, so the
        # UI's SSE contract is preserved without LangChain message plumbing.
        async for event in agent.astream({"query": q.text, "autopilot": q.autopilot}, stream_mode="custom"):
            yield f"data: {json.dumps(event)}\n\n"
        yield sse("total", ms=round((time.perf_counter() - t_start) * 1000))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- UI ---

@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = """<!DOCTYPE html>
<html data-theme="dark"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>K8s Autoscaling Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root[data-theme="dark"]{--bg:#0d1117;--panel:#0f1117;--surface:#161b22;--surface2:#1c2230;--fg:#e6edf3;--border:#2a3038;--muted:#8b949e;--accent:#1f6feb;--accent-hover:#388bfd;--code-bg:#0d1117;--strong:#f0f6fc;--user-bg:#1f6feb;--cpu:#3fb950;--cpu-bg:rgba(63,185,80,0.12);--llm:#a371f7;--llm-bg:rgba(163,113,247,0.14)}
:root[data-theme="light"]{--bg:#f6f8fa;--panel:#ffffff;--surface:#ffffff;--surface2:#f6f8fa;--fg:#1f2328;--border:#d1d9e0;--muted:#656d76;--accent:#0969da;--accent-hover:#0550ae;--code-bg:#f6f8fa;--strong:#1f2328;--user-bg:#0969da;--cpu:#1a7f37;--cpu-bg:rgba(26,127,55,0.10);--llm:#8250df;--llm-bg:rgba(130,80,223,0.10)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column;transition:background .2s,color .2s}
.appbar{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--border)}
.brand{display:flex;align-items:center;gap:12px}
.brand .logo{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--cpu));display:flex;align-items:center;justify-content:center;font-size:16px}
.brand h1{font-size:15px;font-weight:650}
.brand .sub{font-size:12px;color:var(--muted)}
.appbar-right{display:flex;align-items:center;gap:10px}
.tally{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:600;color:var(--cpu);background:var(--cpu-bg);border:1px solid var(--cpu);padding:5px 11px;border-radius:20px}
.tally .dot{width:7px;height:7px;border-radius:50%;background:var(--cpu)}
.iconbtn{background:transparent;border:1px solid var(--border);color:var(--muted);padding:6px 11px;border-radius:7px;cursor:pointer;font-size:12px}
.iconbtn:hover{border-color:var(--accent);color:var(--accent)}
.layout{flex:1;display:grid;grid-template-columns:1fr 390px;min-height:0}
.conv{display:flex;flex-direction:column;min-height:0;border-right:1px solid var(--border)}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:14px}
.empty{margin:auto;text-align:center;color:var(--muted);max-width:400px}
.empty h2{font-size:16px;color:var(--fg);margin-bottom:6px;font-weight:600}
.empty p{font-size:13px;line-height:1.6}
.msg{max-width:88%;padding:11px 15px;border-radius:13px;font-size:14px;line-height:1.7}
.msg.user{align-self:flex-end;background:var(--user-bg);color:#fff;white-space:pre-wrap;border-bottom-right-radius:4px}
.msg.system{align-self:flex-start;background:var(--surface);border:1px solid var(--border);border-bottom-left-radius:4px}
.msg.status{align-self:center;color:var(--muted);font-size:12px;font-style:italic;padding:2px}
.badge{align-self:flex-start;font-size:11px;padding:4px 11px;border-radius:20px;font-weight:600;border:1px solid;display:inline-flex;gap:6px;align-items:center}
.badge.cpu{color:var(--cpu);border-color:var(--cpu);background:var(--cpu-bg)}
.badge.llm{color:var(--llm);border-color:var(--llm);background:var(--llm-bg)}
.msg h1,.msg h2,.msg h3{margin:12px 0 6px;font-weight:600}
.msg h1{font-size:16px}.msg h2{font-size:15px}.msg h3{font-size:14px}
.msg p{margin:6px 0}.msg ul,.msg ol{margin:6px 0;padding-left:20px}.msg li{margin:3px 0}
.msg pre{background:var(--code-bg);padding:12px;border-radius:6px;overflow-x:auto;margin:8px 0;font-size:12.5px;border:1px solid var(--border)}
.msg code{font-size:12.5px;font-family:'SF Mono',Menlo,monospace;background:var(--code-bg);padding:2px 5px;border-radius:3px}
.msg pre code{background:none;padding:0}
.msg strong{color:var(--strong)}.msg hr{border:none;border-top:1px solid var(--border);margin:12px 0}
.scenarios{display:flex;flex-wrap:wrap;gap:8px;padding:10px 20px 0}
.chip{background:var(--surface2);border:1px solid var(--border);color:var(--fg);font-size:12px;padding:6px 12px;border-radius:20px;cursor:pointer;transition:.15s}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.input-area{padding:12px 20px 18px;display:flex;gap:10px}
.input-area textarea{flex:1;background:var(--surface);border:1px solid var(--border);color:var(--fg);padding:11px 13px;border-radius:10px;font-size:14px;resize:none;height:64px;font-family:inherit}
.input-area textarea:focus{outline:none;border-color:var(--accent)}
.input-area button{background:var(--accent);color:#fff;border:none;padding:0 22px;border-radius:10px;cursor:pointer;font-size:14px;font-weight:600}
.input-area button:hover{background:var(--accent-hover)}
.input-area button:disabled{opacity:.5;cursor:not-allowed}
.trace-pane{display:flex;flex-direction:column;min-height:0;background:var(--panel)}
.trace-head{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;font-size:13px;font-weight:600}
.trace-head .total{font-size:12px;color:var(--muted);font-weight:600;font-variant-numeric:tabular-nums}
.trace{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px}
.trace .hint{color:var(--muted);font-size:12px;text-align:center;margin:auto;line-height:1.6;max-width:260px}
.step{display:flex;gap:10px;background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--muted);border-radius:8px;padding:9px 11px;animation:slidein .2s ease}
.step.cpu{border-left-color:var(--cpu)}
.step.llm{border-left-color:var(--llm)}
.step .ic{font-size:15px;line-height:1.4;flex-shrink:0;width:20px;text-align:center}
.step .body{flex:1;min-width:0}
.step .top{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
.step .nm{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.step .dur{font-size:12px;font-variant-numeric:tabular-nums;color:var(--muted);flex-shrink:0}
.step.done.cpu .dur{color:var(--cpu)}
.step.done.llm .dur{color:var(--llm)}
.step .dt{font-size:11.5px;color:var(--muted);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.step .bar{height:3px;border-radius:2px;background:var(--border);margin-top:7px;overflow:hidden}
.step .bar i{display:block;height:100%;width:0;background:var(--cpu);transition:width .3s}
.step.llm .bar i{background:var(--llm)}
.step .spin{width:13px;height:13px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0;align-self:center}
.step .chk{color:var(--cpu);font-weight:700;flex-shrink:0;align-self:center}
.step.llm .chk{color:var(--llm)}
.legend{border-top:1px solid var(--border);padding:10px 16px;display:flex;flex-wrap:wrap;gap:14px;font-size:11px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend .sw{width:9px;height:9px;border-radius:2px}
.gmap{padding:14px 12px 12px;border-bottom:1px solid var(--border);background:var(--panel)}
.gmap .grow{display:flex;align-items:center;justify-content:center;gap:6px}
.gnode{font-size:11px;font-weight:600;padding:5px 10px;border-radius:7px;border:1px solid var(--border);background:var(--surface);color:var(--muted);white-space:nowrap;transition:border-color .2s,color .2s,background .2s}
.gnode.cpu.active,.gnode.cpu.done{border-color:var(--cpu);color:var(--cpu);background:var(--cpu-bg)}
.gnode.llm.active,.gnode.llm.done{border-color:var(--llm);color:var(--llm);background:var(--llm-bg)}
.gnode.active{animation:gpulse 1.1s infinite}
.gnode .gchk{margin-left:5px}
.gconn{width:1px;height:9px;background:var(--border);margin:0 auto}
.gbranch{font-size:11px;color:var(--muted)}
.gloop{font-size:10px;color:var(--muted)}
@keyframes gpulse{0%,100%{opacity:1}50%{opacity:.5}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes slidein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@media(max-width:860px){.layout{grid-template-columns:1fr}.trace-pane{display:none}}
</style></head><body>
<div class="appbar">
  <div class="brand">
    <div class="logo">&#9881;</div>
    <div><h1>K8s Autoscaling Agent</h1><div class="sub">CPU-first agent &middot; right tool for the right task</div></div>
  </div>
  <div class="appbar-right">
    <div class="tally" title="Queries answered without escalating to the LLM"><span class="dot"></span><span id="tallyText">&mdash; handled on CPU</span></div>
    <button class="iconbtn" onclick="toggleTheme()">Theme</button>
    <button class="iconbtn" onclick="clearAll()">Clear</button>
  </div>
</div>
<div class="layout">
  <section class="conv">
    <div class="chat" id="chat"><div class="empty" id="empty"><h2>Ask the agent</h2><p>Paste a Kubernetes autoscaling config or ask a question. Watch the agent route, use read-only tools, retrieve, answer, and check itself &mdash; live on the right.</p></div></div>
    <div class="scenarios" id="scenarios"></div>
    <div class="input-area">
      <textarea id="input" placeholder="Paste a K8s config or ask a question..."></textarea>
      <button id="send" onclick="send()">Send</button>
    </div>
  </section>
  <aside class="trace-pane">
    <div class="trace-head"><span>Agent trace</span><span class="total" id="traceTotal"></span></div>
    <div class="gmap" id="gmap">
      <div class="grow"><div class="gnode cpu" id="g-triage">Triage</div></div>
      <div class="gconn"></div>
      <div class="grow"><div class="gnode cpu" id="g-plan">Plan</div><span class="gbranch">&rarr;</span><div class="gnode cpu" id="g-tools">Tools &#8635;</div></div>
      <div class="gconn"></div>
      <div class="grow"><div class="gnode cpu" id="g-retrieve">Retrieve &middot; Rerank</div></div>
      <div class="gconn"></div>
      <div class="grow"><div class="gnode cpu" id="g-generate">Auditor SLM</div><span class="gbranch">&rarr;</span><div class="gnode llm" id="g-llm">LLM</div></div>
      <div class="gconn"></div>
      <div class="grow"><div class="gnode cpu" id="g-critic">Critic</div><span class="gloop">&#8635; refine</span></div>
    </div>
    <div class="trace" id="trace"><div class="hint">The agent's steps appear here as it works &mdash; each node, its timing, and whether it ran on CPU or the LLM.</div></div>
    <div class="legend"><span><span class="sw" style="background:var(--cpu)"></span>CPU</span><span><span class="sw" style="background:var(--llm)"></span>LLM</span></div>
  </aside>
</div>
<script>
const chat = document.getElementById('chat');
const trace = document.getElementById('trace');
const input = document.getElementById('input');
const btn = document.getElementById('send');
const tallyText = document.getElementById('tallyText');
const traceTotal = document.getElementById('traceTotal');

const NODE = {
  classify:{ic:'\\uD83E\\uDDED'}, plan:{ic:'\\uD83E\\uDDE9'}, extract:{ic:'\\u2702\\uFE0F'},
  tool:{ic:'\\uD83D\\uDD27'}, retrieve:{ic:'\\uD83D\\uDD0E'}, generate:{ic:'\\uD83E\\uDDE0'},
  llm:{ic:'\\u2601\\uFE0F'}, critic:{ic:'\\uD83D\\uDEE1\\uFE0F'}, refine:{ic:'\\u21BB'}, other:{ic:'\\u2022'}
};
function nodeType(name){
  const n = name.toLowerCase();
  if(n.includes('triage')) return 'classify';
  if(n.includes('router')||n.includes('planner')||n.includes('plan')) return 'plan';
  if(n.includes('extractor')||n.includes('extract')) return 'extract';
  if(n.startsWith('tool')) return 'tool';
  if(n.includes('retriever')||n.includes('opensearch')||n.includes('reranker')) return 'retrieve';
  if(n.includes('auditor')) return 'generate';
  if(n.includes('llm')||n.includes('bedrock')) return 'llm';
  if(n.includes('critic')) return 'critic';
  if(n.includes('refine')) return 'refine';
  return 'other';
}
const BAR_SCALE = 3000;
function fmt(ms){ return ms>=1000 ? (ms/1000).toFixed(2)+'s' : ms+' ms'; }
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function renderMd(t){ try{ return marked.parse(t); }catch(e){ return t; } }
function toggleTheme(){ const h=document.documentElement; h.dataset.theme = h.dataset.theme==='dark'?'light':'dark'; }

let totalRuns=0, cpuRuns=0;
function updateTally(){ tallyText.textContent = totalRuns ? (cpuRuns+' / '+totalRuns+' handled on CPU') : '\\u2014 handled on CPU'; }

function addMsg(html, cls){
  const e=document.getElementById('empty'); if(e) e.remove();
  const d=document.createElement('div'); d.className='msg '+cls; d.innerHTML=html;
  chat.appendChild(d); chat.scrollTop=chat.scrollHeight; return d;
}

let steps={};
function addStep(name, note){
  const t = nodeType(name), cpu = t!=='llm';
  const d=document.createElement('div');
  d.className='step running '+(cpu?'cpu':'llm');
  d.innerHTML='<span class="ic">'+(NODE[t]||NODE.other).ic+'</span>'+
    '<div class="body"><div class="top"><span class="nm">'+esc(name)+'</span><span class="dur"></span></div>'+
    '<div class="dt">'+esc(note||'working...')+'</div><div class="bar"><i></i></div></div>'+
    '<span class="spin"></span>';
  trace.appendChild(d); trace.scrollTop=trace.scrollHeight;
  steps[name]=d;
}
function finishStep(name, ms, detail){
  const d=steps[name]; if(!d) return;
  d.classList.remove('running'); d.classList.add('done');
  const sp=d.querySelector('.spin'); if(sp){ const c=document.createElement('span'); c.className='chk'; c.textContent='\\u2713'; sp.replaceWith(c); }
  if(detail) d.querySelector('.dt').textContent=detail;
  d.querySelector('.dur').textContent=fmt(ms);
  d.querySelector('.bar i').style.width=Math.max(3,Math.min(100,ms/BAR_SCALE*100))+'%';
}
function traceNote(text){
  const d=document.createElement('div'); d.className='step'; d.style.borderLeftColor='var(--accent)';
  d.innerHTML='<span class="ic">\\u21BB</span><div class="body"><div class="top"><span class="nm">'+esc(text)+'</span></div></div>';
  trace.appendChild(d); trace.scrollTop=trace.scrollHeight;
}

const GIDS=['triage','plan','tools','retrieve','generate','llm','critic'];
const GMAP={classify:'triage',plan:'plan',extract:'tools',tool:'tools',retrieve:'retrieve',generate:'generate',llm:'llm',critic:'critic'};
function graphEl(name){ const id=GMAP[nodeType(name)]; return id?document.getElementById('g-'+id):null; }
function gReset(){ GIDS.forEach(id=>{ const el=document.getElementById('g-'+id); if(el){ el.classList.remove('active','done'); const c=el.querySelector('.gchk'); if(c) c.remove(); } }); }
function gActivate(name){ const el=graphEl(name); if(el){ el.classList.remove('done'); el.classList.add('active'); } }
function gDone(name){ const el=graphEl(name); if(!el) return; el.classList.remove('active'); el.classList.add('done'); if(!el.querySelector('.gchk')){ const c=document.createElement('span'); c.className='gchk'; c.textContent='\\u2713'; el.appendChild(c); } }

function clearAll(){
  chat.innerHTML='<div class="empty" id="empty"><h2>Ask the agent</h2><p>Paste a Kubernetes autoscaling config or ask a question.</p></div>';
  trace.innerHTML='<div class="hint">The agent\\'s steps appear here as it works.</div>';
  traceTotal.textContent='';
}

const SCENARIOS=[
  {label:'\\uD83D\\uDD27 Live tool use', text:'Why is NodePool `default` not launching nodes?'},
  {label:'\\uD83D\\uDCD8 Concept question', text:'Explain how HPA stabilization windows work'},
  {label:'\\u26A0\\uFE0F Deprecated config', text:'apiVersion: karpenter.sh/v1beta1\\nkind: NodePool\\nmetadata:\\n  name: demo\\nspec:\\n  template:\\n    spec:\\n      requirements:\\n        - key: karpenter.k8s.aws/instance-category\\n          operator: In\\n          values: ["c","m"]'},
  {label:'\\uD83D\\uDEAB Off-topic', text:'what is the weather like today in Seattle?'}
];
const scBar=document.getElementById('scenarios');
SCENARIOS.forEach(s=>{ const b=document.createElement('button'); b.className='chip'; b.textContent=s.label; b.onclick=()=>{ input.value=s.text; send(); }; scBar.appendChild(b); });

async function send(){
  const text=input.value.trim(); if(!text) return;
  input.value=''; btn.disabled=true; steps={};
  trace.innerHTML=''; traceTotal.textContent=''; gReset();
  addMsg(esc(text).replace(/\\n/g,'<br>'),'user');

  let usedLLM=false;
  const resp=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const reader=resp.body.getReader(); const dec=new TextDecoder();
  let responseDiv=null, rawText='', buffer='';

  while(true){
    const {done,value}=await reader.read(); if(done) break;
    buffer+=dec.decode(value,{stream:true});
    const lines=buffer.split('\\n'); buffer=lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: ')) continue;
      const data=line.slice(6);
      if(data==='[DONE]'){
        if(responseDiv&&rawText) responseDiv.innerHTML=renderMd(rawText);
        totalRuns++; if(!usedLLM) cpuRuns++; updateTally();
        break;
      }
      try{
        const msg=JSON.parse(data);
        if(msg.type==='step_start'){ addStep(msg.name, msg.note); gActivate(msg.name); }
        else if(msg.type==='step_done'){ finishStep(msg.name, msg.ms, msg.detail); gDone(msg.name); }
        else if(msg.type==='total'){ traceTotal.textContent='total '+fmt(msg.ms); }
        else if(msg.type==='model'){
          const llm=!msg.name.toLowerCase().includes('slm'); if(llm) usedLLM=true;
          addMsg('<span class="ic">'+(llm?'\\u2601\\uFE0F':'\\uD83E\\uDDE0')+'</span> '+esc(msg.name),'badge '+(llm?'llm':'cpu'));
        }
        else if(msg.type==='response'){ addMsg(renderMd(msg.text),'system'); }
        else if(msg.type==='answer_reset'){
          if(responseDiv&&rawText) responseDiv.innerHTML=renderMd(rawText);
          responseDiv=null; rawText='';
          const esc_note = msg.reason==='escalating'
            ? 'CPU critic kept failing \\u2014 escalating to the LLM.'
            : 'Draft was weakly grounded \\u2014 refining and retrying.';
          addMsg(esc_note,'status');
          traceNote(msg.reason==='escalating'?'escalating to LLM':'refining & retrying');
        }
        else if(msg.type==='token'){
          if(!responseDiv) responseDiv=addMsg('','system');
          rawText+=msg.text; responseDiv.innerHTML=renderMd(rawText); chat.scrollTop=chat.scrollHeight;
        }
      }catch(e){}
    }
  }
  btn.disabled=false; input.focus();
}
input.addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); send(); }});
updateTally();
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
