"""Read-only Kubernetes evidence tools.

Every tool talks to the API via the typed/dynamic Python client with structured
arguments — never a shelled-out kubectl with interpolated user text — so there is
no shell-injection surface. Names/namespaces from the (untrusted) extractor are
validated against the k8s name regex before any call. The orchestrator's RBAC
grants only get/list/watch.
"""
import yaml
from kubernetes import client as k8s_client, config as k8s_config
from kubernetes.dynamic import DynamicClient

from . import config, extract
from .validation import validate_config

_k8s_available = False


def init_k8s():
    """Load cluster credentials once (in-cluster first, then local kubeconfig)."""
    global _k8s_available
    if not config.TOOLS_ENABLED:
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


def available() -> bool:
    return _k8s_available


def _prune(v):
    """Drop null/empty fields so the trimmed view is signal-dense (the client's
    to_dict() emits hundreds of nulls that bury the meaningful fields)."""
    if isinstance(v, dict):
        return {k: pv for k, pv in ((k, _prune(val)) for k, val in v.items())
                if pv not in (None, {}, [], "")}
    if isinstance(v, list):
        return [pv for pv in (_prune(x) for x in v) if pv not in (None, {}, [], "")]
    return v


def _trim_k8s_object(obj: dict) -> str:
    """Keep the fields that explain a resource's behavior; drop server noise."""
    meta = obj.get("metadata", {}) or {}
    trimmed = {
        "kind": obj.get("kind"),
        "metadata": {"name": meta.get("name"), "namespace": meta.get("namespace")},
        "spec": _prune(obj.get("spec") or {}),
    }
    status = obj.get("status") or {}
    if isinstance(status, dict):
        keep = _prune({k: status.get(k) for k in ("conditions", "phase", "reason", "message", "resources")})
        if keep:
            trimmed["status"] = keep
    return yaml.safe_dump(trimmed, default_flow_style=False, sort_keys=False)[:4000]


def describe_resource(args: dict) -> str:
    """GET a single cluster resource and return a trimmed view (read-only)."""
    if not _k8s_available:
        return "Cluster tools are disabled; answering from documentation only."
    name = args.get("name")
    if not extract.valid_k8s_name(name or ""):
        return "No valid resource name found in the query."
    ns = args.get("namespace")
    if ns and not extract.valid_k8s_name(ns):
        return "Invalid namespace."
    try:
        dyn = DynamicClient(k8s_client.ApiClient())
        res = dyn.resources.get(api_version=args["api_version"], kind=args["kind"])
        obj = res.get(name=name, namespace=ns or "default") if args.get("namespaced") else res.get(name=name)
        return _trim_k8s_object(obj.to_dict())
    except Exception as e:
        return f"Could not fetch {args.get('kind')}/{name}: {e}"


def list_events(args: dict) -> str:
    """List recent events, optionally for one object (read-only)."""
    if not _k8s_available:
        return "Cluster tools are disabled; answering from documentation only."
    name = args.get("name")
    if name and not extract.valid_k8s_name(name):
        return "Invalid object name."
    ns = args.get("namespace") or "default"
    if not extract.valid_k8s_name(ns):
        return "Invalid namespace."
    try:
        v1 = k8s_client.CoreV1Api()
        field_selector = f"involvedObject.name={name}" if name else None
        events = v1.list_namespaced_event(namespace=ns, field_selector=field_selector, limit=25)
        items = events.items or []
        if not items:
            return f"No recent events for {name or f'namespace {ns}'}."
        return "\n".join(
            f"{e.last_timestamp or e.event_time} {e.type}/{e.reason}: {(e.message or '').strip()}"
            for e in items[-10:])
    except Exception as e:
        return f"Could not list events: {e}"


def _pod_scheduling_summary(pod) -> str:
    spec = pod.spec
    sel = dict(spec.node_selector) if spec.node_selector else None
    has_affinity = bool(spec.affinity and spec.affinity.node_affinity)
    tols = [t.key for t in (spec.tolerations or []) if getattr(t, "key", None)]
    return (f"    nodeSelector: {sel or 'none'}; "
            f"nodeAffinity: {'set' if has_affinity else 'none'}; tolerations: {tols or 'none'}")


def _latest_warning_event(v1, namespace: str, name: str):
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


def _format_nodepool(d: dict) -> str:
    name = (d.get("metadata") or {}).get("name")
    reqs = (((d.get("spec") or {}).get("template") or {}).get("spec") or {}).get("requirements") or []
    fields = {r.get("key"): r.get("values") for r in reqs if isinstance(r, dict)}
    disruption = ((d.get("spec") or {}).get("disruption") or {}).get("consolidationPolicy")
    return (f"- {name}: capacity-type={fields.get('karpenter.sh/capacity-type') or 'any'}, "
            f"instance-family={fields.get('karpenter.k8s.aws/instance-family') or '-'}, "
            f"instance-category={fields.get('karpenter.k8s.aws/instance-category') or '-'}, "
            f"arch={fields.get('kubernetes.io/arch') or 'any'}, consolidation={disruption or '-'}")


def _nodepool_summaries() -> str:
    try:
        dyn = DynamicClient(k8s_client.ApiClient())
        items = dyn.resources.get(api_version="karpenter.sh/v1", kind="NodePool").get().items
    except Exception as e:
        return f"(could not list NodePools: {e})"
    out = [_format_nodepool(np.to_dict()) for np in items]
    return "\n".join(out) if out else "(no NodePools found)"


def investigate_cluster(args: dict) -> str:
    """Read-only first-look triage: survey unhealthy pods (+ events) and NodePool
    provisioning config. General SRE triage, not problem-specific logic."""
    if not _k8s_available:
        return "Cluster tools are disabled; answering from documentation only."
    namespace = args.get("namespace")
    if namespace and not extract.valid_k8s_name(namespace):
        return "Invalid namespace."
    query = args.get("query") or ""

    # A specifically named NodePool: focus on it. If not found (name may have been
    # mis-extracted, or it doesn't exist), fall through to general triage rather
    # than abort, so we still report real pod health.
    ref = extract.resource_ref(query)
    name = extract.extract_name(query)
    if ref and ref.get("kind") == "NodePool" and name:
        try:
            np = k8s_client.CustomObjectsApi().get_cluster_custom_object(
                "karpenter.sh", "v1", "nodepools", name)
            return "NODEPOOL:\n" + _format_nodepool(np)
        except Exception:
            pass

    v1 = k8s_client.CoreV1Api()
    try:
        pods = (v1.list_namespaced_pod(namespace).items if namespace
                else v1.list_pod_for_all_namespaces().items)
    except Exception as e:
        return f"(could not list pods: {e})\n\nNODEPOOLS:\n" + _nodepool_summaries()

    problems = [p for p in pods
                if p.status.phase == "Pending"
                or (p.status.phase == "Running" and not (p.status.container_statuses
                    and all(c.ready for c in p.status.container_statuses)))]
    scope = f"namespace {namespace}" if namespace else "the cluster"
    if problems:
        lines = [f"{len(problems)} pod(s) not healthy in {scope}:"]
        for p in problems[:5]:
            lines.append(f"  {p.metadata.namespace}/{p.metadata.name} [{p.status.phase}]")
            lines.append(_pod_scheduling_summary(p))
            event = _latest_warning_event(v1, p.metadata.namespace, p.metadata.name)
            if event:
                lines.append(f"    event: {event}")
        return "PROBLEM PODS:\n" + "\n".join(lines)

    # No unhealthy pods. Say so explicitly so the answer can state the reported
    # symptom isn't present rather than invent a cause for it.
    return f"No pending or unhealthy pods found in {scope}.\n\nNODEPOOLS:\n" + _nodepool_summaries()


_TOOLS = {
    "describe_resource": describe_resource,
    "list_events": list_events,
    "validate_config": validate_config,
    "investigate_cluster": investigate_cluster,
}


def run_tool(tool: str, args: dict) -> str:
    fn = _TOOLS.get(tool)
    return fn(args) if fn else f"Unknown tool: {tool}"


def tool_detail(output: str) -> str:
    return (output or "").strip().split("\n")[0][:80]


def format_tool_results(results: list) -> str:
    parts = []
    for r in results:
        parts.append(f"[tool: {r['tool']} \u00b7 {extract.args_summary(r['tool'], r['args'])}]\n{r['output']}")
    return "\n\n".join(parts)
