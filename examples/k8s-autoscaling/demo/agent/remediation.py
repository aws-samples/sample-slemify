"""Gated, bounded write actions (human-in-the-loop).

Apply is OFF unless ALLOW_APPLY is set AND the orchestrator has write RBAC. Each
remediation is a deterministic, validated patch against a SPECIFIC named target
(never free-form model YAML, never a blanket change): it dry-runs server-side
first, touches only the intended field, and is followed by a re-read to verify.
"""
from kubernetes import client as k8s_client

from . import config, extract


def _nodepool_capacity_types(np: dict):
    reqs = (((np.get("spec") or {}).get("template") or {}).get("spec") or {}).get("requirements") or []
    for r in reqs:
        if r.get("key") == "karpenter.sh/capacity-type":
            return r.get("values")
    return None


def enable_spot_on_nodepool(name: str) -> dict:
    """Add 'spot' to one named NodePool's capacity-type. Dry-runs first, changes
    only the capacity-type values."""
    if not config.ALLOW_APPLY:
        return {"ok": False, "message": "Apply is disabled (set ALLOW_APPLY=true and grant write RBAC)."}
    if not extract.valid_k8s_name(name):
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
    api = k8s_client.CustomObjectsApi()
    try:
        np = api.get_cluster_custom_object("karpenter.sh", "v1", "nodepools", name)
    except Exception as e:
        return {"ok": False, "message": f"Could not re-read NodePool {name}: {e}"}
    caps = _nodepool_capacity_types(np)
    if caps and "spot" in caps:
        return {"ok": True, "message": f"Verified: NodePool {name} capacity-type is now {caps}."}
    return {"ok": False, "message": f"Verification failed: capacity-type is {caps}."}


def _all_node_label_keys() -> set:
    keys = set()
    for n in k8s_client.CoreV1Api().list_node().items:
        keys |= set((n.metadata.labels or {}).keys())
    return keys


def _deployment_nodeselector(dep: dict) -> dict:
    return (((dep.get("spec") or {}).get("template") or {}).get("spec") or {}).get("node_selector") or {}


def _unschedulable_nodeselector_keys(dep: dict, node_keys: set) -> list:
    """nodeSelector keys on the Deployment that NO node carries — an impossible
    constraint that pins pods to Pending. Keys present on some node are left
    alone (Karpenter may still provision a matching node)."""
    return sorted(k for k in _deployment_nodeselector(dep) if k not in node_keys)


def fix_unschedulable_nodeselector(target: str) -> dict:
    """Remove the impossible nodeSelector key(s) from a Deployment ('ns/name').
    Drops only keys no node satisfies, dry-runs first, then re-reads to verify."""
    if not config.ALLOW_APPLY:
        return {"ok": False, "message": "Apply is disabled (set ALLOW_APPLY=true and grant write RBAC)."}
    ns, _, name = target.partition("/")
    if not (extract.valid_k8s_name(ns) and extract.valid_k8s_name(name)):
        return {"ok": False, "message": "Invalid Deployment target."}
    apps = k8s_client.AppsV1Api()
    try:
        dep = apps.read_namespaced_deployment(name, ns).to_dict()
    except Exception as e:
        return {"ok": False, "message": f"Could not fetch Deployment {ns}/{name}: {e}"}
    if not _deployment_nodeselector(dep):
        return {"ok": False, "message": f"Deployment {ns}/{name} has no nodeSelector to amend."}
    try:
        bad = _unschedulable_nodeselector_keys(dep, _all_node_label_keys())
    except Exception as e:
        return {"ok": False, "message": f"Could not list nodes: {e}"}
    if not bad:
        return {"ok": True, "message": f"Deployment {ns}/{name} nodeSelector is satisfiable; no change needed."}
    # null value deletes a key under merge patch, so we drop ONLY the impossible
    # keys and leave any valid ones intact.
    patch = {"spec": {"template": {"spec": {"nodeSelector": {k: None for k in bad}}}}}
    try:
        apps.patch_namespaced_deployment(name, ns, body=patch, dry_run="All")
        apps.patch_namespaced_deployment(name, ns, body=patch)
    except Exception as e:
        return {"ok": False, "message": f"Apply failed (dry-run or apply): {e}"}
    return {"ok": True, "message": f"Removed unschedulable nodeSelector key(s) {bad} from {ns}/{name}; pods can now schedule."}


def verify_deployment_schedulable(target: str) -> dict:
    ns, _, name = target.partition("/")
    apps = k8s_client.AppsV1Api()
    try:
        dep = apps.read_namespaced_deployment(name, ns).to_dict()
    except Exception as e:
        return {"ok": False, "message": f"Could not re-read Deployment {ns}/{name}: {e}"}
    try:
        bad = _unschedulable_nodeselector_keys(dep, _all_node_label_keys())
    except Exception as e:
        return {"ok": False, "message": f"Could not list nodes: {e}"}
    if bad:
        return {"ok": False, "message": f"Verification failed: unsatisfiable nodeSelector key(s) still present: {bad}."}
    selector = _deployment_nodeselector(dep)
    return {"ok": True, "message": f"Verified: {ns}/{name} nodeSelector is now satisfiable (keys: {sorted(selector) or 'none'})."}


# Whitelist of structured remediations: action -> (apply, verify), single target.
REMEDIATIONS = {
    "enable_spot_on_nodepool": (enable_spot_on_nodepool, verify_nodepool_spot),
    "fix_unschedulable_nodeselector": (fix_unschedulable_nodeselector, verify_deployment_schedulable),
}


def detect_remediation(query: str) -> dict | None:
    """Map a query to a safe remediation on an EXPLICITLY NAMED target that
    genuinely has the problem. Returns None otherwise — apply is always bounded
    to a resource the user pointed at, never a blanket change."""
    if not config.ALLOW_APPLY:
        return None
    ref = extract.resource_ref(query)
    name = extract.extract_name(query)
    if not (ref and name):
        return None
    kind = ref.get("kind")
    if kind == "NodePool":
        try:
            np = k8s_client.CustomObjectsApi().get_cluster_custom_object(
                "karpenter.sh", "v1", "nodepools", name)
        except Exception:
            return None
        caps = _nodepool_capacity_types(np)
        if caps is not None and "spot" not in caps:
            return {"action": "enable_spot_on_nodepool", "target": name,
                    "summary": f"add 'spot' to NodePool {name} capacity-type",
                    "manual": (f"kubectl edit nodepool {name}\n"
                               "# under spec.template.spec.requirements, add \"spot\" to the\n"
                               "# values of the karpenter.sh/capacity-type requirement")}
        return None
    if kind == "Deployment":
        ns = extract.extract_namespace(query) or "default"
        if not extract.valid_k8s_name(ns):
            return None
        try:
            dep = k8s_client.AppsV1Api().read_namespaced_deployment(name, ns).to_dict()
            bad = _unschedulable_nodeselector_keys(dep, _all_node_label_keys())
        except Exception:
            return None
        if bad:
            keys = ", ".join(bad)
            return {"action": "fix_unschedulable_nodeselector", "target": f"{ns}/{name}",
                    "summary": f"remove impossible nodeSelector key(s) [{keys}] from Deployment {ns}/{name}",
                    "manual": (f"kubectl edit deployment {name} -n {ns}\n"
                               f"# remove the unschedulable nodeSelector key(s) [{keys}]\n"
                               "# from spec.template.spec.nodeSelector")}
        return None
    return None
