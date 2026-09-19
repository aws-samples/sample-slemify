#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# One query against the orchestrator, asserting the stream carries a cost event
# with at least one Bedrock call and a total event. Used by seed.sh and by the
# skip-ahead scripts to confirm the demo answers before handing the account to
# an attendee. The first query is run here so an attendee's is never the first.
#
# Overridable: NAMESPACE, LOCAL_PORT, QUERY.

set -euo pipefail

NAMESPACE="${NAMESPACE:-slemify}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
QUERY="${QUERY:-My Karpenter NodePool sets limits.cpu to 50 and pods are stuck Pending at 48 CPU. Is that the cause?}"

kubectl port-forward -n "${NAMESPACE}" svc/k8s-autoscaling-orchestrator "${LOCAL_PORT}:80" >/dev/null 2>&1 &
PF_PID=$!
trap 'kill "${PF_PID}" 2>/dev/null || true' EXIT
sleep 4

# Wait for /health to report ready (warmup can take a few seconds).
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:${LOCAL_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

body="$(jq -nc --arg t "${QUERY}" '{text: $t, autopilot: false}')"
stream="$(curl -sN -X POST "http://localhost:${LOCAL_PORT}/query" \
  -H 'Content-Type: application/json' \
  -d "${body}" --max-time 300)"

echo "--- stream tail ---"
printf '%s\n' "${stream}" | tail -n 8

if ! printf '%s' "${stream}" | grep -q '"type": *"cost"' && \
   ! printf '%s' "${stream}" | grep -q '"cost"'; then
  echo "SMOKE FAIL: no cost event in the stream" >&2
  exit 1
fi
if ! printf '%s' "${stream}" | grep -q 'total'; then
  echo "SMOKE FAIL: no total event in the stream" >&2
  exit 1
fi

echo "SMOKE OK: cost and total events present"
