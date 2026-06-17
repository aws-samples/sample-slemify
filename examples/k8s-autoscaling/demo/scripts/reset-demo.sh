#!/bin/bash
set -euo pipefail

# Reset the remediation demo scenarios back to their broken state, so you can
# show the agent fixing them again. Safe to run repeatedly.
#
#   ./scripts/reset-demo.sh          # re-break both scenarios
#   ./scripts/reset-demo.sh --clean  # remove the demo resources entirely
#
# What it does:
#   - Spot cost: re-applies the NodePool with capacity-type on-demand-only,
#     dropping any "spot" the agent added.
#   - Pending pods: recreates the Deployment so all replicas come back freshly
#     Pending (a plain apply would leave a rolling mix of old Running pods).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$(dirname "$SCRIPT_DIR")/scenarios"
NAMESPACE="slemify"

NODEPOOL_MANIFEST="${SCENARIOS_DIR}/spot-cost-nodepool.yaml"
DEPLOY_MANIFEST="${SCENARIOS_DIR}/pending-pods-deployment.yaml"

if [[ "${1:-}" == "--clean" ]]; then
  echo "=== Removing demo scenario resources ==="
  kubectl delete -f "${DEPLOY_MANIFEST}" --ignore-not-found
  kubectl delete -f "${NODEPOOL_MANIFEST}" --ignore-not-found
  echo "Done. Demo scenarios removed."
  exit 0
fi

echo "=== Resetting: Spot-cost NodePool (on-demand only) ==="
kubectl apply -f "${NODEPOOL_MANIFEST}"

echo "=== Resetting: pending-pods Deployment (recreate so all replicas are Pending) ==="
kubectl delete deployment payments-api -n "${NAMESPACE}" --ignore-not-found --wait=true
kubectl apply -f "${DEPLOY_MANIFEST}"

echo
echo "=== Broken state restored ==="
kubectl get nodepool demo-spot-misconfigured \
  -o jsonpath='NodePool demo-spot-misconfigured capacity-type: {.spec.template.spec.requirements[?(@.key=="karpenter.sh/capacity-type")].values}{"\n"}'
kubectl get pods -n "${NAMESPACE}" -l app=payments-api \
  -o custom-columns=POD:.metadata.name,STATUS:.status.phase --no-headers || true
echo
echo "Both scenarios are broken again and ready to demo."
