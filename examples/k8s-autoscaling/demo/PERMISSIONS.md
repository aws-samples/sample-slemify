# Agent Permissions

Everything this agent can ever do to your cluster, in one place. Read this
before turning on `ALLOW_APPLY`, and re-read it whenever `agent/patch_schema.py`
or `k8s-rbac-apply.yaml` changes — they are two halves of one authority
boundary and this document is how you check they still agree.

## The two halves of the boundary

The agent can only mutate the cluster if **both** of these say yes to the same
change. Either one alone is not enough:

1. **RBAC (who): `k8s-rbac-apply.yaml`.** Kubernetes-enforced. Grants the
   sandbox tools ServiceAccount — the *only* workload with cluster credentials
   — `get`/`patch`/`update` on exactly two resource **kinds**: `NodePool` and
   `Deployment`. No `create`, no `delete`, on anything, ever. This RBAC is
   opt-in and not applied by default.
2. **Patch schema (what): `agent/patch_schema.py:PATCH_SCHEMA`.** Code-enforced,
   inside the sandbox pod, on every single write. Grants specific **fields** on
   those kinds, each with its own value constraints. RBAC says "may patch a
   NodePool"; the schema says "and only `spec.limits.cpu`, `spec.limits.memory`,
   or the capacity-type list, never anything else on that object."

RBAC is coarse (kind-level) and can't express "only this field." The schema is
what actually narrows "can patch a NodePool" down to "can only raise its CPU/
memory limit or add a capacity type." **The schema is doing most of the real
work here** — treat RBAC as the outer fence and the schema as the actual lock.

## Exactly what can be changed today

| Kind | Field | Verb | Constraint | What it's for |
|------|-------|------|------------|----------------|
| `NodePool` | `spec.limits.cpu` | set a quantity | ≥ `1` | Unblock provisioning when the limit is `0` |
| `NodePool` | `spec.limits.memory` | set a quantity | ≥ `1Gi` | Same, for memory |
| `NodePool` | capacity-type requirement | add to a fixed set | value ∈ `{spot, on-demand, reserved}` | Allow Spot capacity |
| `Deployment` | `spec.template.spec.nodeSelector` | remove specific keys | only keys **no node in the cluster carries** | Un-stick pods pinned to an impossible selector |
| `Deployment` | `spec.replicas` | set an integer | `0`–`50` | Change desired replica count |

This table is generated from `PATCH_SCHEMA` by
`scripts/print-permissions.py` (see below) — if the two ever disagree, the
script output is the truth, this table is documentation that can go stale.

Every entry above resolves to one of a small, fixed set of **verbs** in
`patch_schema.py` (`set_quantity`, `set_int`, `add_to_set`,
`remove_unschedulable_nodeselector_keys`). A verb is generic mechanism (how to
validate + build + verify a patch shape); a schema entry is one field wired to
one verb. Adding a new *field* to an existing verb is a one-line schema entry.
Adding a genuinely new *kind of write* means writing a new verb — which is more
code, but still a single reusable function, not a scenario.

## What can never happen, structurally

- **No field outside the table above, ever** — not because the model was told
  not to, but because `remediation.plan_patch`/`apply_patch` reject any
  `(kind, field)` pair not in `PATCH_SCHEMA` before doing anything else, and
  the model's own proposal is separately constrained by a JSON Schema
  (`patch_schema.json_schema_for_kind`) passed to the CPU auditor SLM as
  `response_format` — llama.cpp compiles that into a grammar and masks the
  sampler, so the model cannot even *emit* a field name outside the list.
- **No `create` or `delete`**, of anything — not in RBAC, not in any verb.
- **No free-form YAML, ever** — the model never produces a manifest or a
  kubectl command that gets executed. It proposes `{field, value}`; the
  patch body is always built by code from the schema, never from model text.
- **No blanket changes** — every proposal targets one resource the user named
  in their query (validated as a real Kubernetes name before any API call);
  there is no "fix every NodePool" path.
- **No mutation without a dry-run first** — every apply calls the K8s API with
  `dry_run="All"` before the real patch, for every verb, unconditionally.
- **No apply without `ALLOW_APPLY=true` and the RBAC above granted** — absent
  either one, the agent is `get`/`list`/`watch` only.

## Autopilot vs. approve

Both modes go through the exact same validation, dry-run, and verify steps.
The only difference is who clicks "go":

- **Approve** (`ALLOW_APPLY=true`, autopilot off): the agent computes and
  validates the patch, shows you the summary and the exact `kubectl patch`
  command that would reproduce it, and waits for you to click *Apply this fix*.
- **Autopilot** (approve + the per-request toggle): the same validated,
  dry-run patch is applied immediately, then re-read to verify.

Turning autopilot on does **not** widen what can be changed — it only removes
the human click on an already-narrow, already-validated set of possible writes.

## Verifying this yourself

Three checks, none of which require reading Python if you don't want to:

```bash
# 1. What kinds/verbs does RBAC actually grant the sandbox ServiceAccount?
kubectl get clusterrole k8s-autoscaling-tools-apply -o yaml

# 2. Is the write path even turned on right now?
kubectl get deployment k8s-autoscaling-tools -n slemify \
  -o jsonpath='{.spec.template.spec.containers[0].env}' | grep -o 'ALLOW_APPLY[^,}]*'

# 3. What fields/constraints does the schema currently allow, in plain text
#    (generated straight from PATCH_SCHEMA, not hand-maintained)?
python3 scripts/print-permissions.py
```

If (3)'s output ever lists a kind that (1)'s RBAC doesn't grant `patch` on,
that field is dead code — the schema would allow it but RBAC would reject it
at the API server. If it's the other way around (RBAC grants a kind with no
matching schema entries), that kind is currently un-patchable by design; it's
not a gap, just headroom RBAC leaves for a future schema entry.
