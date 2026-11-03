#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build the tuned knowledge index against the retriever the attendee just
# fine-tuned. The vectors an index stores must come from the same encoder that
# embeds the query at search time, so this index can only be built once the
# tuned retriever is serving (module 2). It writes the k8s-autoscaling-knowledge
# index that `make recall` scores as the "slemify" and "slemify+rerank" modes,
# and that EMBED=slemify reads at query time.
#
# Idempotent: re-running rebuilds the index. On the workshop IDE it reuses the
# forwards the IDE service keeps open on 9200 and 8083. Elsewhere it starts its
# own temporary forwards to OpenSearch and the retriever.
set -euo pipefail

NAMESPACE="${NAMESPACE:-slemify}"
INDEX_NAME="${INDEX_NAME:-k8s-autoscaling-knowledge}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building the tuned knowledge index ($INDEX_NAME) against the fine-tuned retriever..."

# The retriever must be serving before its encoder can embed the corpus.
kubectl rollout status deployment/k8s-autoscaling-retriever-inference \
  -n "$NAMESPACE" --timeout=300s

if command -v systemctl >/dev/null 2>&1 &&
  systemctl cat slemify-forward@.service >/dev/null 2>&1; then
  # The IDE service holds 9200 and 8083, so a second forward would fail with
  # "address already in use". Wait for its forwards instead. The retriever was
  # just rolled out, so its forward may need one restart cycle (about 30 s).
  for port in 9200 8083; do
    ok=""
    for _ in $(seq 1 45); do
      code="$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$port/" || true)"
      if [ -n "$code" ] && [ "$code" != "000" ]; then ok=1; break; fi
      sleep 2
    done
    if [ -z "$ok" ]; then
      echo "The forward on $port is not answering. Run make forward, then try again." >&2
      exit 1
    fi
  done
else
  kubectl port-forward --address 127.0.0.1 -n "$NAMESPACE" \
    svc/opensearch-cluster-master 9200:9200 >/dev/null 2>&1 &
  PF_OS_PID=$!
  kubectl port-forward --address 127.0.0.1 -n "$NAMESPACE" \
    svc/k8s-autoscaling-retriever-inference 8083:8080 >/dev/null 2>&1 &
  PF_RET_PID=$!
  trap 'kill "$PF_OS_PID" "$PF_RET_PID" 2>/dev/null || true' EXIT
  sleep 4
fi

INDEX_NAME="$INDEX_NAME" EMBEDDING_URL="http://localhost:8083" \
  python3 "$HERE/index-knowledge.py" --embedder=slemify

echo "Done. 'make recall' and EMBED=slemify now read $INDEX_NAME."
