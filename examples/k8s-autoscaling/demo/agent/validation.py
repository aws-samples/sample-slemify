"""Deterministic, client-side config validation (no cluster, no API).

Used two ways: as the `validate_config` tool (lint a pasted manifest) and as the
critic's draft-fix lint (catch a proposed fix that uses a deprecated apiVersion
before it's shown or applied). Being deterministic, it's a reliable signal the
model can ground on — unlike anything heuristic.
"""
import re

import yaml

# Deprecated apiVersions the validator flags, mapped to the current one.
DEPRECATED_API_VERSIONS = {
    "autoscaling/v2beta1": "autoscaling/v2",
    "autoscaling/v2beta2": "autoscaling/v2",
    "karpenter.sh/v1alpha5": "karpenter.sh/v1",
    "karpenter.sh/v1beta1": "karpenter.sh/v1",
    "karpenter.k8s.aws/v1beta1": "karpenter.k8s.aws/v1",
    "policy/v1beta1": "policy/v1",
    "extensions/v1beta1": "apps/v1 (or networking.k8s.io/v1 for Ingress)",
}


def validate_config(args: dict) -> str:
    """Structural + deprecation lint of pasted YAML. Server-side dry-run would
    need write RBAC, so this stays purely local: parse, flag missing required
    fields and deprecated apiVersions."""
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
                f"{label}: apiVersion '{api_version}' is deprecated; use '{DEPRECATED_API_VERSIONS[api_version]}'")
    if not findings:
        return "No structural or deprecation issues found (client-side checks)."
    return "; ".join(findings)


def validate_draft_fix(draft: str) -> list:
    """Lint any YAML the draft proposes, so the agent catches a fix that uses
    stale/invalid config before showing or applying it — grounding the fix, not
    just the diagnosis."""
    issues = []
    for block in re.findall(r"```(?:ya?ml)?\s*\n(.*?)```", draft or "", re.DOTALL):
        if "apiVersion" in block and "kind" in block:
            result = validate_config({"yaml": block})
            if result and not result.startswith("No structural"):
                issues.append(result)
    return issues
