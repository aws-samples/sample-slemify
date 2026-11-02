#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Keep one workshop port-forward alive. Installed on the workshop IDE as
# /usr/local/bin/slemify-forward and run by slemify-forward@<port>.service.
#
#   slemify-forward <local-port>
#
# Looks the port up in forwards.conf, waits until the Service has a ready pod,
# then runs `kubectl port-forward` bound to 127.0.0.1 and checks it over HTTP.
# It exits non-zero when the forward exits or stops answering, and systemd
# starts it again. Services get redeployed during the lab, so the pod behind
# a forward changes; this is how the forward follows it.
#
# Exit codes: 0 stopped, 1 restart me, 2 port not in forwards.conf (the unit
# sets RestartPreventExitStatus=2 so a typo does not loop).
set -u

PORT="${1:-}"
CONF="${FORWARDS_CONF:-/etc/slemify/forwards.conf}"
KUBECTL="${KUBECTL:-kubectl}"
NAMESPACE="${NAMESPACE:-slemify}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-10}"
HEALTH_FAILS="${HEALTH_FAILS:-3}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10}"
ENDPOINT_POLL="${ENDPOINT_POLL:-10}"

log() { echo "slemify-forward[$PORT]: $*"; }

row=""
if [ -n "$PORT" ] && [ -r "$CONF" ]; then
  row="$(awk -v p="$PORT" '!/^[[:space:]]*#/ && NF >= 5 && $1 == p { print $2, $3, $4; exit }' "$CONF")"
fi
if [ -z "$row" ]; then
  log "port '$PORT' is not in $CONF"
  exit 2
fi
read -r SVC RPORT HPATH <<<"$row"

# The IDE bootstrap writes the kubeconfig once the cluster exists. If this
# starts first, write it here; while the cluster is still creating, retry.
KCFG="${KUBECONFIG:-$HOME/.kube/config}"
if [ ! -s "$KCFG" ]; then
  log "no kubeconfig at $KCFG yet, running aws eks update-kubeconfig"
  if ! aws eks update-kubeconfig --region "${AWS_REGION:-}" \
    --name "${EKS_CLUSTER_NAME:-}" --kubeconfig "$KCFG" >/dev/null 2>&1; then
    log "cluster not reachable yet, retrying in 15 s"
    sleep 15
    exit 1
  fi
fi

has_ready_pod() {
  "$KUBECTL" get endpointslices -n "$NAMESPACE" \
    -l "kubernetes.io/service-name=$SVC" \
    -o jsonpath='{range .items[*].endpoints[*]}{.conditions.ready}{" "}{end}' 2>/dev/null |
    grep -q true
}

# Wait for a ready pod. This covers a model not deployed yet and a Service
# scaled to zero (the re-ranker before module 2). Log once, not every poll.
waited=""
until has_ready_pod; do
  if [ -z "$waited" ]; then
    log "waiting for a ready pod behind svc/$SVC"
    waited=1
  fi
  sleep "$ENDPOINT_POLL"
done

CHILD=""
cleanup() {
  if [ -n "$CHILD" ]; then
    # CONT too, so a stopped child still gets the TERM.
    kill -TERM "$CHILD" 2>/dev/null || true
    kill -CONT "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 0' TERM INT

log "forwarding 127.0.0.1:$PORT to svc/$SVC:$RPORT"
"$KUBECTL" port-forward --address 127.0.0.1 -n "$NAMESPACE" \
  "svc/$SVC" "$PORT:$RPORT" &
CHILD=$!

# Any HTTP code counts as up (the same rule as make forward-check), so no
# model needs a health path. Only "no answer" (000) counts as a failure, and
# it takes HEALTH_FAILS in a row, so a slow reply from a busy model does not
# restart the forward.
fails=0
while true; do
  sleep "$HEALTH_INTERVAL" &
  wait $!
  if ! kill -0 "$CHILD" 2>/dev/null; then
    wait "$CHILD"
    rc=$?
    CHILD=""
    log "port-forward exited (code $rc), restarting"
    exit 1
  fi
  code="$(curl -s -o /dev/null -m "$HEALTH_TIMEOUT" -w '%{http_code}' \
    "http://127.0.0.1:$PORT$HPATH" 2>/dev/null)"
  if [ -z "$code" ] || [ "$code" = "000" ]; then
    fails=$((fails + 1))
    log "no answer on 127.0.0.1:$PORT$HPATH ($fails of $HEALTH_FAILS)"
    if [ "$fails" -ge "$HEALTH_FAILS" ]; then
      log "forward not answering, restarting"
      exit 1
    fi
  else
    fails=0
  fi
done
