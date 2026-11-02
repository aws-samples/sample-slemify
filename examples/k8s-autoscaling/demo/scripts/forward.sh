#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Start or check the lab's port-forwards. Behind `make forward` and
# `make forward-check`. The port table is ../forwards.conf.
#
#   forward.sh up [port...]   (re)start the forwards, then print the check
#   forward.sh check          print the state of each forward
#
# Two modes:
#   Supervised (the workshop IDE): a systemd service (slemify-forward@<port>)
#     keeps every forward alive and restarts it when it drops. `up` asks
#     systemd to restart the service instead of starting a second forward.
#   Manual (a laptop, or any host without that service): `up` starts each
#     forward itself with kubectl, detached. Core forwards always start; a
#     model forward starts only once its Service has a ready pod.
# Every forward binds to 127.0.0.1 only.
#
# Must run on macOS bash 3.2: no mapfile, no associative arrays.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${FORWARDS_CONF:-$HERE/../forwards.conf}"
NAMESPACE="${NAMESPACE:-slemify}"
KUBECTL="${KUBECTL:-kubectl}"

# Print the port table without comments or blank lines.
rows() {
  grep -v '^[[:space:]]*#' "$CONF" | awk 'NF >= 5 { print $1, $2, $3, $4, $5 }'
}

supervised() {
  command -v systemctl >/dev/null 2>&1 &&
    systemctl cat slemify-forward@.service >/dev/null 2>&1
}

# True when the Service has at least one ready endpoint (a running pod).
has_ready_pod() {
  "$KUBECTL" get endpointslices -n "$NAMESPACE" \
    -l "kubernetes.io/service-name=$1" \
    -o jsonpath='{range .items[*].endpoints[*]}{.conditions.ready}{" "}{end}' 2>/dev/null |
    grep -q true
}

http_code() {
  curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$1$2" 2>/dev/null
}

# True when the port is in the argument list, or the list is empty.
wanted() {
  local p="$1"; shift
  [ "$#" -eq 0 ] && return 0
  local w
  for w in "$@"; do [ "$w" = "$p" ] && return 0; done
  return 1
}

# A row is expected to be up when it is core, or a model with a ready pod.
expected() {
  [ "$1" = "core" ] || has_ready_pod "$2"
}

cmd_check() {
  if supervised; then
    echo "The IDE keeps these forwards open. Details: systemctl status slemify-forward@<port>,"
    echo "logs: journalctl -u slemify-forward@<port>"
  fi
  local lp svc rp path kind code state
  rows | while read -r lp svc rp path kind; do
    expected "$kind" "$svc" || continue
    code="$(http_code "$lp" "$path")"
    if [ -z "$code" ] || [ "$code" = "000" ]; then
      state="DOWN (not reachable)"
    else
      state="up (HTTP $code)"
    fi
    printf "%-5s %-36s %s\n" "$lp" "$svc" "$state"
  done
}

# Wait up to 20 s for the given ports (all when none) to answer.
wait_up() {
  local _ lp svc rp path kind pending code
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    pending=0
    while read -r lp svc rp path kind; do
      wanted "$lp" "$@" || continue
      expected "$kind" "$svc" || continue
      code="$(http_code "$lp" "$path")"
      if [ -z "$code" ] || [ "$code" = "000" ]; then pending=1; fi
    done <<EOF
$(rows)
EOF
    [ "$pending" -eq 0 ] && return 0
    sleep 2
  done
  return 0
}

up_supervised() {
  local p
  if [ "$#" -eq 0 ]; then
    sudo -n systemctl restart slemify-forwards.target
  else
    for p in "$@"; do
      sudo -n systemctl restart "slemify-forward@$p.service"
    done
  fi
}

start_forward() {
  local lp="$1" svc="$2" rp="$3"
  # Kill a stale forward on this local port first, so two never share it.
  pkill -f "port-forward.*$lp:$rp" >/dev/null 2>&1 || true
  sleep 1
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup "$KUBECTL" port-forward --address 127.0.0.1 -n "$NAMESPACE" \
      "svc/$svc" "$lp:$rp" >"/tmp/pf-$lp.log" 2>&1 </dev/null &
  else
    nohup "$KUBECTL" port-forward --address 127.0.0.1 -n "$NAMESPACE" \
      "svc/$svc" "$lp:$rp" >"/tmp/pf-$lp.log" 2>&1 </dev/null &
  fi
  disown 2>/dev/null || true
  echo "forwarding $lp -> $svc"
}

up_manual() {
  local lp svc rp path kind
  while read -r lp svc rp path kind; do
    wanted "$lp" "$@" || continue
    expected "$kind" "$svc" || continue
    start_forward "$lp" "$svc" "$rp"
  done <<EOF
$(rows)
EOF
}

cmd_up() {
  if supervised; then
    up_supervised "$@" || return 1
  else
    up_manual "$@"
  fi
  wait_up "$@"
  cmd_check
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  check) cmd_check ;;
  *) echo "usage: $0 up [port...] | check" >&2; exit 2 ;;
esac
