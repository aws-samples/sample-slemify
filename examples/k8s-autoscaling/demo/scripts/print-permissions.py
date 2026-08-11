#!/usr/bin/env python3
"""Print exactly what agent/patch_schema.py allows the agent to write, in
plain text, generated straight from PATCH_SCHEMA (never hand-maintained).

Run from this directory: python3 scripts/print-permissions.py
See PERMISSIONS.md for what this is checking and why.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import patch_schema as ps  # noqa: E402


def main():
    print("=== Agent write permissions (from agent/patch_schema.py) ===\n")
    if not ps.PATCH_SCHEMA:
        print("(PATCH_SCHEMA is empty: the agent cannot patch anything.)")
        return
    for kind, fields in ps.PATCH_SCHEMA.items():
        print(f"{kind}:")
        for field, spec in fields.items():
            verb = spec["verb"]
            constraints = []
            if "min" in spec:
                constraints.append(f"min={spec['min']}")
            if "max" in spec:
                constraints.append(f"max={spec['max']}")
            if "allowed_values" in spec:
                constraints.append(f"allowed={spec['allowed_values']}")
            constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
            desc = spec.get("description", "")
            print(f"  - {field}  (verb: {verb}){constraint_str}")
            if desc:
                print(f"      {desc}")
        print()
    print("Verbs available (mechanism only; a verb with no schema entry above")
    print("cannot be triggered by anything):")
    for verb in ps.VERBS:
        print(f"  - {verb}")
    print()
    print("Nothing outside the fields listed above can ever be patched, and no")
    print("kind outside this list can ever be patched, regardless of RBAC. Cross")
    print("-check against RBAC with:")
    print("  kubectl get clusterrole k8s-autoscaling-tools-apply -o yaml")


if __name__ == "__main__":
    main()
