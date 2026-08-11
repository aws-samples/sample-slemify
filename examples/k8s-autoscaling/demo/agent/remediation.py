"""Generic, gated, bounded write actions (human-in-the-loop).

Apply is OFF unless ALLOW_APPLY is set AND the orchestrator has write RBAC.

Unlike a hand-written function per failure scenario, this module has exactly
ONE apply/verify pair: apply_patch/verify_patch. What field of what kind may be
touched, and with what value, is entirely determined by patch_schema.PATCH_SCHEMA
plus the verb-level validation it defines (a range, an enum, a quantity parse).
Onboarding a new fixable field is a schema entry, not a new function.

The write itself is still deterministic and inspectable: dry-run server-side
first, patch, re-read to verify — the same discipline the old per-scenario
functions had, just generalized. What CAN reach this path (which proposal is
worth making) is decided upstream, either by detect_remediation's evidence-based
heuristics or by the model's schema-constrained proposal (see graph.n_propose_fix);
this module only ever executes a proposal that already validated against the
schema — it never trusts a field/value it hasn't checked itself.
"""
from kubernetes import client as k8s_client

from . import config, extract, patch_schema
from .patch_schema import PATCH_SCHEMA, VERBS, get_nested


def _split_target(kind: str, target: str):
    """'ns/name' for namespaced kinds, 'name' for cluster-scoped."""
    if "/" in target:
        ns, _, name = target.partition("/")
        return ns, name
    return None, target


def _get_resource(kind: str, name: str, namespace: str | None) -> dict | None:
    if kind == "NodePool":
        return k8s_client.CustomObjectsApi().get_cluster_custom_object(
            "karpenter.sh", "v1", "nodepools", name)
    if kind == "Deployment":
        return k8s_client.AppsV1Api().read_namespaced_deployment(name, namespace or "default").to_dict()
    raise ValueError(f"unsupported kind: {kind}")


def _patch_resource(kind: str, name: str, namespace: str | None, patch: dict, dry_run: bool):
    kwargs = {"dry_run": "All"} if dry_run else {}
    if kind == "NodePool":
        k8s_client.CustomObjectsApi().patch_cluster_custom_object(
            "karpenter.sh", "v1", "nodepools", name, body=patch, **kwargs)
    elif kind == "Deployment":
        k8s_client.AppsV1Api().patch_namespaced_deployment(
            name, namespace or "default", body=patch, **kwargs)
    else:
        raise ValueError(f"unsupported kind: {kind}")


def _gather_aux(kind: str, field: str, spec: dict) -> dict:
    """Extra read-only evidence a verb needs beyond the resource itself. Kept
    to a tiny, explicit set (today: node label keys, for the nodeSelector
    verb) rather than a generic side-channel, so what gets read stays visible."""
    if spec["verb"] == "remove_unschedulable_nodeselector_keys":
        keys = set()
        for n in k8s_client.CoreV1Api().list_node().items:
            keys |= set((n.metadata.labels or {}).keys())
        return {"node_label_keys": keys}
    return {}


def plan_patch(kind: str, target: str, field: str, value: str) -> dict:
    """Validate + compute the patch and its human-readable summary, without
    touching the cluster's write path (still reads the resource + evidence)."""
    ok, norm_value, err = patch_schema.validate_proposal(kind, field, value)
    if not ok:
        return {"ok": False, "message": f"Rejected: {err}"}
    ns, name = _split_target(kind, target)
    if not extract.valid_k8s_name(name):
        return {"ok": False, "message": "Invalid resource name."}
    if ns and not extract.valid_k8s_name(ns):
        return {"ok": False, "message": "Invalid namespace."}
    spec = PATCH_SCHEMA[kind][field]
    verb = VERBS[spec["verb"]]
    try:
        current = _get_resource(kind, name, ns)
    except Exception as e:
        return {"ok": False, "message": f"Could not fetch {kind}/{name}: {e}"}
    try:
        aux = _gather_aux(kind, field, spec)
    except Exception as e:
        return {"ok": False, "message": f"Could not gather evidence for {kind}/{name}: {e}"}
    plan = verb["plan_patch"](current, field, spec, norm_value, aux)
    plan.update(kind=kind, name=name, namespace=ns, field=field, value=norm_value, aux=aux)
    return {"ok": True, **plan}


def apply_patch(kind: str, target: str, field: str, value: str) -> dict:
    """Dry-run then apply a schema-validated patch to one named resource."""
    if not config.ALLOW_APPLY:
        return {"ok": False, "message": "Apply is disabled (set ALLOW_APPLY=true and grant write RBAC)."}
    plan = plan_patch(kind, target, field, value)
    if not plan["ok"]:
        return plan
    if plan.get("noop"):
        return {"ok": True, "message": plan["message"]}
    try:
        _patch_resource(kind, plan["name"], plan["namespace"], plan["patch"], dry_run=True)
        _patch_resource(kind, plan["name"], plan["namespace"], plan["patch"], dry_run=False)
    except Exception as e:
        return {"ok": False, "message": f"Apply failed (dry-run or apply): {e}"}
    return {"ok": True, "message": plan["message"]}


def verify_patch(kind: str, target: str, field: str, value: str) -> dict:
    ns, name = _split_target(kind, target)
    spec = PATCH_SCHEMA.get(kind, {}).get(field)
    if not spec:
        return {"ok": False, "message": f"{field!r} is not an allowed field for {kind}."}
    verb = VERBS[spec["verb"]]
    try:
        current_after = _get_resource(kind, name, ns)
        aux = _gather_aux(kind, field, spec)
    except Exception as e:
        return {"ok": False, "message": f"Could not re-read {kind}/{name}: {e}"}
    ok, message = verb["verify"](current_after, field, spec, value, aux)
    return {"ok": ok, "message": ("Verified: " if ok else "Verification failed: ") + message}


def manual_command_for(kind: str, target: str, field: str, value: str) -> str:
    """The kubectl command that reproduces the exact patch plan_patch computed,
    for the 'run it yourself' text shown when autopilot is off."""
    plan = plan_patch(kind, target, field, value)
    if not plan.get("ok") or plan.get("noop") or not plan.get("patch"):
        return ""
    return patch_schema.manual_command(kind, plan["name"], plan.get("namespace"), plan["patch"])


# --- Evidence-based detection (no model call) ---
# Kept from the previous implementation: a query naming a resource that
# genuinely, checkably has one of the known bad states is offered a fix
# immediately, without waiting on a model round-trip. This is a starting
# point, not the only path — graph.n_propose_fix additionally asks the model
# to propose a schema-constrained fix for cases this heuristic doesn't cover.


def _nodepool_capacity_types(np: dict):
    reqs = get_nested(np, "spec.template.spec.requirements") or []
    for r in reqs:
        if r.get("key") == "karpenter.sh/capacity-type":
            return r.get("values")
    return None


def _all_node_label_keys() -> set:
    keys = set()
    for n in k8s_client.CoreV1Api().list_node().items:
        keys |= set((n.metadata.labels or {}).keys())
    return keys


def named_target(query: str) -> dict | None:
    """Cheap parse only (no cluster read): does the query name a resource of a
    kind the schema can ever patch? detect_remediation already covers the
    known-shape, no-model-call fast path; this exists only to decide whether
    it is worth spending a model call proposing a schema-constrained fix for
    everything else. Gated the same as detect_remediation: no write path is
    surfaced at all unless ALLOW_APPLY is set."""
    if not config.ALLOW_APPLY:
        return None
    ref = extract.resource_ref(query)
    name = extract.extract_name(query)
    if not (ref and name):
        return None
    kind = ref.get("kind")
    if kind not in PATCH_SCHEMA:
        return None
    if kind == "Deployment":
        ns = extract.extract_namespace(query) or "default"
        if not extract.valid_k8s_name(ns):
            return None
        return {"kind": kind, "target": f"{ns}/{name}"}
    return {"kind": kind, "target": name}


def detect_remediation(query: str) -> dict | None:
    """Cheap, evidence-checked heuristic for the two known-shape problems this
    demo ships scenarios for. Returns a proposal dict {kind, target, field,
    value, summary, manual} or None. Bounded to a resource the user explicitly
    named, and only offered if that resource genuinely has the problem."""
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
        limit = get_nested(np, "spec.limits.cpu")
        if str(limit) == "0":
            return _proposal("NodePool", name, "spec.limits.cpu", "100")
        caps = _nodepool_capacity_types(np)
        if caps is not None and "spot" not in caps:
            return _proposal("NodePool", name, "capacity_type", "spot")
        return None
    if kind == "Deployment":
        ns = extract.extract_namespace(query) or "default"
        if not extract.valid_k8s_name(ns):
            return None
        try:
            dep = k8s_client.AppsV1Api().read_namespaced_deployment(name, ns).to_dict()
        except Exception:
            return None
        selector = get_nested(dep, "spec.template.spec.nodeSelector") or {}
        node_keys = _all_node_label_keys()
        if any(k not in node_keys for k in selector):
            return _proposal("Deployment", f"{ns}/{name}", "spec.template.spec.nodeSelector", "")
        return None
    return None


def _proposal(kind: str, target: str, field: str, value: str) -> dict:
    plan = plan_patch(kind, target, field, value)
    summary = plan.get("message") if plan.get("ok") else patch_schema.describe_fix(kind, field, value)
    return {
        "kind": kind, "target": target, "field": field, "value": value,
        "summary": summary,
        "manual": manual_command_for(kind, target, field, value),
    }
