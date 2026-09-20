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

# In freshly vended accounts the first Anthropic invocations can be denied for
# 10 to 15 minutes while the Marketplace entitlement settles ("not authorized
# to perform the required AWS Marketplace actions"). The orchestrator surfaces
# that as a stream that stops before the cost event. Retry across that window
# rather than fail a 30 minute provision on it.
ATTEMPTS="${SMOKE_ATTEMPTS:-12}"
for attempt in $(seq 1 "${ATTEMPTS}"); do
  stream="$(curl -sN -X POST "http://localhost:${LOCAL_PORT}/query" \
    -H 'Content-Type: application/json' \
    -d "${body}" --max-time 300)"
  if printf '%s' "${stream}" | grep -q '"type": *"cost"' && \
     printf '%s' "${stream}" | grep -q '"type": *"total"'; then
    echo "--- stream tail ---"
    printf '%s\n' "${stream}" | tail -n 4
    echo "SMOKE OK: cost and total events present (attempt ${attempt})"
    exit 0
  fi
  echo "attempt ${attempt}/${ATTEMPTS}: stream ended without cost/total; last event:"
  printf '%s\n' "${stream}" | grep '^data:' | tail -n 1 | cut -c1-160
  [ "${attempt}" -lt "${ATTEMPTS}" ] && sleep 60
done

echo "SMOKE FAIL: no complete answer after ${ATTEMPTS} attempts" >&2
exit 1
