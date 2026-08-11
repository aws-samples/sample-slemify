#!/bin/bash
set -euo pipefail

# Reset the remediation demo scenarios back to their broken state, so you can
# show the agent fixing them again. Safe to run repeatedly.
#
#   ./scripts/reset-demo.sh          # re-break both scenarios
#   ./scripts/reset-demo.sh --clean  # remove the demo resources entirely
#
# Spot cost: NodePool capacity-type back to on-demand only.
# Pending pods: NodePool demo-payments cpu limit back to 0, payments-api
#   recreated, and its provisioned node removed so all replicas come back
#   freshly Pending (rather than scheduling onto a leftover node).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$(dirname "$SCRIPT_DIR")/scenarios"
NAMESPACE="slemify"
SPOT="${SCENARIOS_DIR}/spot-cost-nodepool.yaml"
LIMITED="${SCENARIOS_DIR}/limited-nodepool.yaml"

if [[ "${1:-}" == "--clean" ]]; then
  echo "=== Removing demo scenario resources ==="
  kubectl delete -f "${LIMITED}" --ignore-not-found
  kubectl delete -f "${SPOT}" --ignore-not-found
  kubectl delete node -l demo.slemify.io/pool=payments --ignore-not-found
  echo "Done. Demo scenarios removed."
  exit 0
fi

echo "=== Reset: Spot-cost NodePool (on-demand only) ==="
kubectl apply -f "${SPOT}"

echo "=== Reset: pending pods (NodePool limit 0, fresh Pending) ==="
kubectl delete deployment payments-api -n "${NAMESPACE}" --ignore-not-found --wait=true
kubectl delete node -l demo.slemify.io/pool=payments --ignore-not-found
kubectl apply -f "${LIMITED}"
# The agent's fix is a PATCH, which leaves apply's last-applied annotation at 0,
# so `apply` alone may report "unchanged" and not revert a fixed limit. Force it.
kubectl patch nodepool demo-payments --type merge -p '{"spec":{"limits":{"cpu":"0"}}}'

echo
echo "=== Broken state restored ==="
kubectl get nodepool demo-spot-misconfigured \
  -o jsonpath='spot-pool capacity-type: {.spec.template.spec.requirements[?(@.key=="karpenter.sh/capacity-type")].values}{"\n"}'
kubectl get nodepool demo-payments \
  -o jsonpath='payments-pool cpu limit: {.spec.limits.cpu}{"\n"}'
kubectl get pods -n "${NAMESPACE}" -l app=payments-api \
  -o custom-columns=POD:.metadata.name,STATUS:.status.phase --no-headers || true
echo
echo "Both scenarios are broken again and ready to demo."
