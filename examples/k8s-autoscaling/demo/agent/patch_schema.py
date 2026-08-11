"""Generic patch schema: the shape of writes the agent may propose.

This is the single source of truth for what the agent can ever change. Instead
of one hand-written function per demo scenario, a "verb" (set_quantity, set_int,
add_to_set, remove_unschedulable_nodeselector_keys) knows how to validate,
build, and verify a patch to ONE field given a small spec. PATCH_SCHEMA maps
kind -> field -> {verb, constraints, description}. Onboarding a new fixable
field is a schema entry, not a new function; onboarding a new verb (a new KIND
of write, e.g. "toggle a bool") is a small, reusable addition, not a scenario.

The model proposes {field, value} constrained to this schema (a JSON Schema
built from PATCH_SCHEMA is passed as response_format to the CPU auditor SLM, so
it cannot name a field or kind that isn't here). remediation.py does the
cluster IO (fetch/patch/verify); this module is pure Python — no Kubernetes
client — so it can be tested and reasoned about without a cluster.

See PERMISSIONS.md for the human-readable version of exactly what this lets
the agent touch, and scripts/print-permissions.py to check it against live RBAC.
"""
import copy
import json
import re

# --- Quantity parsing (pure) ---
# A deliberately small parser for the K8s quantity suffixes this demo's fields
# use (decimal and binary SI). It resolves a suffix to a float multiplier so
# values can be compared numerically; it is not a full implementation of the
# Kubernetes quantity spec (e.g. no exponent form).
_SUFFIX_MULTIPLIERS = {
    "": 1.0, "n": 1e-9, "u": 1e-6, "m": 1e-3,
    "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18,
    "Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40, "Pi": 2 ** 50, "Ei": 2 ** 60,
}
_QUANTITY_RE = re.compile(r"^([+-]?[0-9]*\.?[0-9]+)([A-Za-z]*)$")


def parse_quantity(value) -> float:
    """Parse a K8s-style quantity ('100m', '2', '1Gi') to a float in its base
    unit (cores for CPU, bytes for memory). Raises ValueError if unparseable."""
    m = _QUANTITY_RE.match(str(value).strip())
    if not m:
        raise ValueError(f"not a valid quantity: {value!r}")
    num, suffix = m.group(1), m.group(2)
    if suffix not in _SUFFIX_MULTIPLIERS:
        raise ValueError(f"unsupported quantity suffix: {suffix!r}")
    return float(num) * _SUFFIX_MULTIPLIERS[suffix]


# --- Nested dict helpers (pure) ---

def get_nested(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def set_nested_patch(path: str, value) -> dict:
    """Build the smallest merge-patch dict that sets one dotted field path."""
    patch = value
    for part in reversed(path.split(".")):
        patch = {part: patch}
    return patch


# --- Verb implementations ---
# Each verb: validate(value, spec) -> (ok, normalized_value, error)
#            plan_patch(current, field, spec, value, aux) -> {noop, patch, message}
#            verify(current_after, field, spec, value, aux) -> (ok, message)
# aux carries cluster evidence a verb needs beyond the resource itself (e.g.
# live node label keys), gathered once by remediation.py per call.

def _validate_quantity(value, spec):
    try:
        v = parse_quantity(value)
    except Exception as e:
        return False, None, str(e)
    if "min" in spec and v < parse_quantity(spec["min"]):
        return False, None, f"must be >= {spec['min']}"
    if "max" in spec and v > parse_quantity(spec["max"]):
        return False, None, f"must be <= {spec['max']}"
    return True, str(value).strip(), None


def _validate_int(value, spec):
    try:
        v = int(str(value).strip())
    except Exception:
        return False, None, "not a valid integer"
    if "min" in spec and v < spec["min"]:
        return False, None, f"must be >= {spec['min']}"
    if "max" in spec and v > spec["max"]:
        return False, None, f"must be <= {spec['max']}"
    return True, v, None


def _validate_add_to_set(value, spec):
    allowed = spec.get("allowed_values") or []
    if value not in allowed:
        return False, None, f"must be one of {allowed}"
    return True, value, None


def _validate_ignored(value, spec):
    return True, value, None


def _plan_set_scalar(current, field, spec, value, aux):
    cur_val = get_nested(current, field)
    return {"noop": False, "patch": set_nested_patch(field, value),
            "message": f"Set {field} to {value} (was {cur_val})."}


def _verify_set_scalar(current_after, field, spec, value, aux):
    cur = get_nested(current_after, field)
    try:
        if spec["verb"] == "set_int":
            ok = int(cur) == int(value)
        else:
            ok = abs(parse_quantity(str(cur)) - parse_quantity(str(value))) < 1e-9
    except Exception:
        ok = str(cur) == str(value)
    return ok, f"{field} is now {cur}."


def _plan_add_to_set(current, field, spec, value, aux):
    reqs = copy.deepcopy(get_nested(current, spec["list_path"]) or [])
    matched = False
    for r in reqs:
        if isinstance(r, dict) and r.get(spec["match_field"]) == spec["match_value"]:
            vals = r.get(spec["target_field"]) or []
            if value in vals:
                return {"noop": True, "patch": None,
                        "message": f"{spec['match_value']} already includes {value!r}; no change needed."}
            r[spec["target_field"]] = sorted(set(vals) | {value})
            matched = True
    if not matched:
        return {"noop": True, "patch": None,
                "message": f"No matching requirement {spec['match_value']!r} found to amend."}
    return {"noop": False, "patch": set_nested_patch(spec["list_path"], reqs),
            "message": f"Added {value!r} to {spec['match_value']}."}


def _verify_add_to_set(current_after, field, spec, value, aux):
    for r in get_nested(current_after, spec["list_path"]) or []:
        if isinstance(r, dict) and r.get(spec["match_field"]) == spec["match_value"]:
            vals = r.get(spec["target_field"]) or []
            return (value in vals), f"{spec['match_value']} values are now {vals}."
    return False, "requirement not found after patch."


def _plan_remove_unschedulable(current, field, spec, value, aux):
    node_keys = aux.get("node_label_keys") or set()
    ns = get_nested(current, field) or {}
    bad = sorted(k for k in ns if k not in node_keys)
    if not bad:
        return {"noop": True, "patch": None, "message": "nodeSelector is already satisfiable; no change needed."}
    return {"noop": False, "patch": set_nested_patch(field, {k: None for k in bad}),
            "message": f"Removed unschedulable key(s) {bad}."}


def _verify_remove_unschedulable(current_after, field, spec, value, aux):
    node_keys = aux.get("node_label_keys") or set()
    ns = get_nested(current_after, field) or {}
    bad = sorted(k for k in ns if k not in node_keys)
    if bad:
        return False, f"still unschedulable key(s): {bad}."
    return True, f"nodeSelector is now satisfiable (keys: {sorted(ns) or 'none'})."


VERBS = {
    "set_quantity": {"validate": _validate_quantity, "plan_patch": _plan_set_scalar, "verify": _verify_set_scalar},
    "set_int": {"validate": _validate_int, "plan_patch": _plan_set_scalar, "verify": _verify_set_scalar},
    "add_to_set": {"validate": _validate_add_to_set, "plan_patch": _plan_add_to_set, "verify": _verify_add_to_set},
    "remove_unschedulable_nodeselector_keys": {
        "validate": _validate_ignored,
        "plan_patch": _plan_remove_unschedulable,
        "verify": _verify_remove_unschedulable,
    },
}


# --- The schema itself: kind -> field -> spec ---
# This table, plus the K8s write RBAC in k8s-rbac-apply.yaml, is the entire
# authority boundary for what the agent can mutate. See PERMISSIONS.md.
PATCH_SCHEMA = {
    "NodePool": {
        "spec.limits.cpu": {
            "verb": "set_quantity", "min": "1",
            "description": "Raise the NodePool's CPU limit so it can provision nodes",
        },
        "spec.limits.memory": {
            # 4Gi, not 1Gi: spec.limits constrains the TOTAL memory of every node
            # the NodePool provisions, not pod requests. No real EC2 instance
            # type has less than a few GiB, so a lower floor lets a
            # technically-valid proposal (e.g. the schema minimum) still leave
            # every instance type "exceeding limits" and pods stuck Pending.
            "verb": "set_quantity", "min": "4Gi",
            "description": "Raise the NodePool's memory limit so it can provision nodes",
        },
        "capacity_type": {
            "verb": "add_to_set",
            "list_path": "spec.template.spec.requirements",
            "match_field": "key", "match_value": "karpenter.sh/capacity-type",
            "target_field": "values", "allowed_values": ["spot", "on-demand", "reserved"],
            "description": "Add a capacity type (spot/on-demand/reserved) to the NodePool's requirements",
        },
    },
    "Deployment": {
        "spec.template.spec.nodeSelector": {
            "verb": "remove_unschedulable_nodeselector_keys",
            "description": "Remove nodeSelector key(s) that no node in the cluster satisfies",
        },
        "spec.replicas": {
            "verb": "set_int", "min": 0, "max": 50,
            "description": "Change the Deployment's desired replica count",
        },
    },
}

# Kubernetes resource name kubectl expects, for the manual command shown to the
# user (falls back to lowercasing the kind if not listed).
KUBECTL_RESOURCE = {"NodePool": "nodepool", "Deployment": "deployment"}


def fields_for_kind(kind: str) -> dict:
    """field -> human description (+ constraints), for the model's prompt."""
    out = {}
    for field, spec in PATCH_SCHEMA.get(kind, {}).items():
        desc = spec.get("description", field)
        constraints = []
        if "min" in spec:
            constraints.append(f"min {spec['min']}")
        if "max" in spec:
            constraints.append(f"max {spec['max']}")
        if "allowed_values" in spec:
            constraints.append(f"one of {spec['allowed_values']}")
        if constraints:
            desc += " (" + ", ".join(constraints) + ")"
        out[field] = desc
    return out


def json_schema_for_kind(kind: str) -> dict:
    """The JSON Schema passed as response_format so the model's proposal is
    structurally constrained to a field that exists for this kind."""
    fields = list(PATCH_SCHEMA.get(kind, {}).keys())
    return {
        "type": "object",
        "properties": {
            "no_fix": {"type": "boolean", "description": "true if no safe, allowed fix applies to this resource"},
            "field": {"type": "string", "enum": fields + ["none"],
                      "description": "which allowed field to change; 'none' if no_fix is true"},
            "value": {"type": "string", "description": "the value to set, as a string; empty if not applicable"},
            "reason": {"type": "string", "description": "one sentence explaining the fix, or why no fix applies"},
        },
        "required": ["no_fix", "field", "value", "reason"],
        "additionalProperties": False,
    }


def validate_proposal(kind: str, field: str, value: str):
    """The one validation entrypoint: is this (kind, field, value) something
    the schema allows, and is the value acceptable for that field's verb?
    Returns (ok, normalized_value, error)."""
    field_spec = PATCH_SCHEMA.get(kind, {}).get(field)
    if not field_spec:
        return False, None, f"{field!r} is not an allowed field for {kind}"
    verb = VERBS.get(field_spec["verb"])
    if not verb:
        return False, None, f"unknown verb {field_spec['verb']!r}"
    return verb["validate"](value, field_spec)


def describe_fix(kind: str, field: str, value: str) -> str:
    spec = PATCH_SCHEMA.get(kind, {}).get(field, {})
    label = spec.get("description", field)
    return f"{label} -> {value}" if value else label


def manual_command(kind: str, name: str, namespace: str | None, patch_body: dict) -> str:
    """kubectl command that reproduces the exact patch a plan/apply computed,
    so 'do it yourself' always matches what autopilot would actually run."""
    resource = KUBECTL_RESOURCE.get(kind, kind.lower())
    ns_flag = f" -n {namespace}" if namespace else ""
    return f"kubectl patch {resource} {name}{ns_flag} --type merge -p '{json.dumps(patch_body)}'"
