"""Extraction and tool routing from the user's message.

Manifest detection (validate path), resource/name/namespace extraction, and the
heuristic that picks which read-only tool to run on the diagnose path. All
outputs are treated as untrusted: names/namespaces are validated against the
k8s name regex before any tool uses them against the API.
"""
import re

# RFC 1123 DNS subdomain. Tool args from the (untrusted) extractor must match
# before any are used against the API server.
_K8S_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")

# Resource keyword -> API metadata, most specific first ("nodepool" before
# "node", "poddisruptionbudget" before "pod"). Read-only kinds only.
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

# Signals of a runtime symptom / cost concern. Used only to choose between a
# targeted describe and a broad investigation; the investigation itself is
# state-driven, not keyed to these words.
_EVENT_SIGNALS = (
    "launch", "pending", "stuck", "fail", "event", "scal", "provision",
    "error", "crash", "evict", "terminat", "disrupt", "not work", "not com",
)
_OPERATIONAL_SIGNALS = _EVENT_SIGNALS + (
    "cost", "expensive", "bill", "spot", "saving", "price", "cheaper",
    "on-demand", "ondemand",
)


def valid_k8s_name(name: str) -> bool:
    return bool(name) and bool(_K8S_NAME_RE.match(name))


def looks_like_yaml(text: str) -> bool:
    """True if the message contains a pasted manifest (apiVersion + kind)."""
    return bool(re.search(r"^\s*apiVersion:\s*\S", text, re.MULTILINE)) and \
        bool(re.search(r"^\s*kind:\s*\S", text, re.MULTILINE))


def extract_manifest(text: str) -> str:
    """Isolate the YAML manifest from a message that wraps it in prose."""
    for block in re.findall(r"```(?:ya?ml)?\s*\n(.*?)```", text, re.DOTALL):
        if "apiVersion" in block and "kind" in block:
            return block.strip()
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"\s*apiVersion:\s*\S", ln)), None)
    if start is None:
        return text.strip()
    block = lines[start:]
    while block:
        s = block[-1].strip()
        if s == "" or block[-1][:1] in (" ", "\t") or s.startswith("- ") or re.match(r"[\w.\-/]+:(\s|$)", s):
            break
        block.pop()
    return "\n".join(block).strip()


def resource_ref(text: str) -> dict | None:
    low = text.lower()
    for keyword, meta in RESOURCE_KEYWORDS:
        if keyword in low:
            return meta
    return None


def extract_name(text: str) -> str | None:
    """A delimited identifier (backticks/quotes) or a 'named/called X' form, only
    if it is a valid k8s name."""
    for pattern in (r"`([^`]+)`", r"\"([^\"]+)\"", r"'([^']+)'",
                    r"\bnamed\s+([a-z0-9][-a-z0-9.]*)", r"\bcalled\s+([a-z0-9][-a-z0-9.]*)"):
        m = re.search(pattern, text)
        if m and valid_k8s_name(m.group(1)):
            return m.group(1)
    return None


def extract_namespace(text: str) -> str | None:
    for pattern in (r"-n\s+`?([a-z0-9][-a-z0-9.]*)", r"--namespace\s+`?([a-z0-9][-a-z0-9.]*)",
                    r"namespace\s+[`'\"]?([a-z0-9][-a-z0-9.]*)",
                    r"\bin\s+the\s+[`'\"]?([a-z0-9][-a-z0-9.]*)[`'\"]?\s+namespace"):
        m = re.search(pattern, text)
        if m and valid_k8s_name(m.group(1)):
            return m.group(1)
    return None


def is_operational(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in _OPERATIONAL_SIGNALS)


def select_cluster_tools(query: str) -> list:
    """On the diagnose path, pick which read-only cluster tool(s) to run: a named
    resource is described directly (with events on a symptom); otherwise a broad
    live investigation. (Manifest linting is handled separately, before this.)"""
    ref = resource_ref(query)
    name = extract_name(query)
    operational = is_operational(query)
    if ref and name:
        return ["describe_resource", "list_events"] if operational else ["describe_resource"]
    return ["investigate_cluster"]


def extract_args(query: str, tool: str) -> dict:
    if tool == "validate_config":
        return {"yaml": extract_manifest(query)}
    if tool == "investigate_cluster":
        return {"namespace": extract_namespace(query), "query": query}
    ref = resource_ref(query) or {}
    return {
        "api_version": ref.get("api_version"),
        "kind": ref.get("kind"),
        "namespaced": ref.get("namespaced", True),
        "name": extract_name(query),
        "namespace": extract_namespace(query),
    }


def args_summary(tool: str, args: dict) -> str:
    if tool == "validate_config":
        return "pasted manifest"
    if tool == "investigate_cluster":
        return f"namespace {args.get('namespace') or 'all'}"
    if tool == "list_events":
        return f"{args.get('name') or '(namespace)'} in {args.get('namespace') or 'default'}"
    return f"{args.get('kind')}/{args.get('name')}"
