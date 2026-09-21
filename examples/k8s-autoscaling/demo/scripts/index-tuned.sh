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
# Idempotent: re-running rebuilds the index. Manages its own port-forwards to
# OpenSearch and the retriever, so it does not depend on forwards being up.
set -euo pipefail

NAMESPACE="${NAMESPACE:-slemify}"
INDEX_NAME="${INDEX_NAME:-k8s-autoscaling-knowledge}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building the tuned knowledge index ($INDEX_NAME) against the fine-tuned retriever..."

# The retriever must be serving before its encoder can embed the corpus.
kubectl rollout status deployment/k8s-autoscaling-retriever-inference \
  -n "$NAMESPACE" --timeout=300s

kubectl port-forward -n "$NAMESPACE" svc/opensearch-cluster-master 9200:9200 >/dev/null 2>&1 &
PF_OS_PID=$!
kubectl port-forward -n "$NAMESPACE" svc/k8s-autoscaling-retriever-inference 8083:8080 >/dev/null 2>&1 &
PF_RET_PID=$!
trap 'kill "$PF_OS_PID" "$PF_RET_PID" 2>/dev/null || true' EXIT
sleep 4

INDEX_NAME="$INDEX_NAME" EMBEDDING_URL="http://localhost:8083" \
  python3 "$HERE/index-knowledge.py" --embedder=slemify

echo "Done. 'make recall' and EMBED=slemify now read $INDEX_NAME."
