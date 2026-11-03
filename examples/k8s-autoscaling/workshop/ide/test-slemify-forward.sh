#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Behavior test for slemify-forward.sh with a stub kubectl. Needs bash,
# python3 and curl; no cluster and no systemd. Runs in under 30 s.
#
#   bash workshop/ide/test-slemify-forward.sh
#
# The stub answers `get endpointslices` with a ready endpoint (or none when
# STUB_NO_ENDPOINTS=1) and serves `port-forward` with a local HTTP server on
# the requested port. It logs every call, so each case also checks that the
# forward binds to 127.0.0.1.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$HERE/slemify-forward.sh"
WORK="$(mktemp -d)"
PORT=18383
PASS=0
FAIL=0

cleanup() {
  pkill -f "http.server $PORT" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$WORK/bin" "$WORK/home/.kube"
echo "stub" >"$WORK/home/.kube/config.stub"
cat >"$WORK/conf" <<EOF
# test table
$PORT test-svc 8080 / model
EOF
cat >"$WORK/bin/kubectl" <<'EOF'
#!/usr/bin/env bash
echo "$*" >>"$STUB_LOG"
case "$*" in
  *"get endpointslices"*)
    [ "${STUB_NO_ENDPOINTS:-0}" = "1" ] || printf 'true '
    ;;
  *port-forward*)
    lp=""
    for a in "$@"; do case "$a" in *:*) lp="${a%%:*}" ;; esac; done
    exec python3 -m http.server "$lp" --bind 127.0.0.1 >/dev/null 2>&1
    ;;
esac
EOF
chmod +x "$WORK/bin/kubectl"

export PATH="$WORK/bin:$PATH"
export FORWARDS_CONF="$WORK/conf"
export KUBECONFIG="$WORK/home/.kube/config.stub"
export HEALTH_INTERVAL=1 HEALTH_FAILS=2 HEALTH_TIMEOUT=1 ENDPOINT_POLL=1
export STUB_LOG="$WORK/log"

ok() { echo "PASS: $*"; PASS=$((PASS + 1)); }
ko() { echo "FAIL: $*"; FAIL=$((FAIL + 1)); }

server_pid() { pgrep -f "http.server $PORT" | head -n 1; }

wait_server() {
  local _
  for _ in $(seq 1 50); do
    [ "$(curl -s -o /dev/null -m 1 -w '%{http_code}' "http://127.0.0.1:$PORT/")" = "200" ] && return 0
    sleep 0.1
  done
  return 1
}

# Wait up to $2 seconds for pid $1 to exit; set rc to its exit code, or to
# "timeout". Runs in this shell (not a subshell) so `wait` sees the child.
wait_exit() {
  local pid="$1" limit="$2" _
  rc=timeout
  for _ in $(seq 1 $((limit * 10))); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      rc=$?
      return 0
    fi
    sleep 0.1
  done
}

check_address() {
  if grep port-forward "$STUB_LOG" | grep -qv -- "--address 127.0.0.1"; then
    ko "$1: a port-forward ran without --address 127.0.0.1"
  else
    ok "$1: every port-forward binds to 127.0.0.1"
  fi
}

# (a) The forward process dies: the supervisor exits non-zero in one interval.
: >"$STUB_LOG"
bash "$SUT" "$PORT" >"$WORK/a.out" 2>&1 &
sup=$!
if wait_server; then
  kill "$(server_pid)"
  wait_exit "$sup" 3
  if [ "$rc" != "timeout" ] && [ "$rc" -ne 0 ]; then
    ok "(a) child exit: supervisor exited with $rc"
  else
    ko "(a) child exit: supervisor result $rc"; kill "$sup" 2>/dev/null
  fi
else
  ko "(a) the stub forward never answered"; kill "$sup" 2>/dev/null
fi
check_address "(a)"

# (b) The forward hangs: after HEALTH_FAILS timeouts the supervisor exits
# non-zero and kills the hung child.
: >"$STUB_LOG"
bash "$SUT" "$PORT" >"$WORK/b.out" 2>&1 &
sup=$!
if wait_server; then
  child="$(server_pid)"
  kill -STOP "$child"
  wait_exit "$sup" 10
  if [ "$rc" != "timeout" ] && [ "$rc" -ne 0 ]; then
    ok "(b) hung forward: supervisor exited with $rc"
  else
    ko "(b) hung forward: supervisor result $rc"; kill "$sup" 2>/dev/null
  fi
  sleep 0.5
  if kill -0 "$child" 2>/dev/null; then
    ko "(b) hung child $child is still running"
    kill -KILL "$child" 2>/dev/null
  else
    ok "(b) hung child was killed"
  fi
else
  ko "(b) the stub forward never answered"; kill "$sup" 2>/dev/null
fi
check_address "(b)"

# (c) No ready pod: no port-forward starts during three polls.
: >"$STUB_LOG"
STUB_NO_ENDPOINTS=1 bash "$SUT" "$PORT" >"$WORK/c.out" 2>&1 &
sup=$!
sleep 3.5
if grep -q port-forward "$STUB_LOG"; then
  ko "(c) a port-forward started with no ready pod"
else
  ok "(c) no port-forward without a ready pod ($(grep -c endpointslices "$STUB_LOG") polls)"
fi
kill "$sup" 2>/dev/null
wait "$sup" 2>/dev/null
check_address "(c)"

# (d) Unknown port: exit code 2.
: >"$STUB_LOG"
bash "$SUT" 19999 >"$WORK/d.out" 2>&1
rc=$?
if [ "$rc" -eq 2 ]; then ok "(d) unknown port: exit 2"; else ko "(d) unknown port: exit $rc"; fi
check_address "(d)"

echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]
