#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Seed the workshop cluster to the state the lab starts in: SLM NodePools present,
# one SLM node pre-warmed, OpenSearch up with the Bedrock (Titan) index the
# monolith reads, and the demo running in monolith seats
# (TRIAGE=off EMBED=bedrock RERANK=off ANALYST=llm GATE=off) with a smoke query
# already run. Idempotent: safe to re-run.
#
# Runs after terraform apply. Reads cluster name, region, bucket, and Bedrock
# region from the terraform outputs unless they are passed as environment
# variables. Assumes kubectl is installed and the caller has cluster-admin
# (terraform grants the apply identity that via enable_cluster_creator_admin_permissions).
#
# Required tools: kubectl, helm, aws, python3, jq.
# Images: set DEMO_IMAGE and RERANKER_IMAGE to pre-built multi-arch images, or
# run demo/scripts/build-images.sh first (needs arm64 + x86 build hosts).

set -euo pipefail

SEED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "${SEED_DIR}")"
TF_DIR="${INFRA_DIR}/terraform"
DEMO_DIR="$(cd "${INFRA_DIR}/../../demo" && pwd)"
NAMESPACE="slemify"

log() { printf '\n=== %s ===\n' "$1"; }

# --- Resolve configuration: env vars first (CodeBuild sets them), terraform
# outputs as the fallback for local runs.
tf_out() {
  if command -v terraform >/dev/null 2>&1; then
    terraform -chdir="${TF_DIR}" output -raw "$1" 2>/dev/null || true
  fi
}

CLUSTER_NAME="${EKS_CLUSTER_NAME:-${CLUSTER_NAME:-$(tf_out cluster_name)}}"
REGION="${AWS_REGION:-$(tf_out region)}"
MODEL_BUCKET="${MODEL_BUCKET:-$(tf_out model_bucket)}"
BEDROCK_REGION="${BEDROCK_REGION:-$(tf_out bedrock_region)}"

if [[ -z "${CLUSTER_NAME}" || -z "${REGION}" ]]; then
  echo "ERROR: could not resolve the cluster name and region." >&2
  echo "Set EKS_CLUSTER_NAME and AWS_REGION, or run 'terraform apply' in ${TF_DIR} first." >&2
  exit 1
fi
BEDROCK_REGION="${BEDROCK_REGION:-${REGION}}"

# Bedrock model IDs. Workshop Studio vended accounts have Bedrock only in
# us-east-1/us-west-2; use a us. inference profile there. Overridable.
LLM_MODEL="${LLM_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
BEDROCK_EMBED_MODEL="${BEDROCK_EMBED_MODEL:-amazon.titan-embed-text-v2:0}"

# Demo images. Default to the account's ECR (built by build-images.sh); override
# for a shared registry.
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY_DEFAULT="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
DEMO_IMAGE="${DEMO_IMAGE:-${REGISTRY_DEFAULT}/slemify/k8s-autoscaling-orchestrator:latest}"
RERANKER_IMAGE="${RERANKER_IMAGE:-${REGISTRY_DEFAULT}/slemify/k8s-autoscaling-reranker:latest}"

echo "Cluster:        ${CLUSTER_NAME} (${REGION})"
echo "Model bucket:   ${MODEL_BUCKET}"
echo "Bedrock region: ${BEDROCK_REGION}"
echo "LLM model:      ${LLM_MODEL}"
echo "Demo image:     ${DEMO_IMAGE}"

# --- kubectl ---
log "Configuring kubectl"
aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${REGION}" >/dev/null
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# --- SLM NodePools + pre-warm ---
log "Applying SLM NodePools"
kubectl apply -f "${SEED_DIR}/slm-nodepools.yaml"

log "Pre-warming one SLM node"
kubectl apply -f "${SEED_DIR}/prewarm.yaml"
# Auto Mode takes a minute or two to launch a node; do not block the rest of the
# seed on it, but surface progress.
kubectl rollout status deployment/slm-prewarm -n "${NAMESPACE}" --timeout=300s || \
  echo "  (pre-warm still pending; continuing — Auto Mode will catch up)"

# --- OpenSearch ---
log "Deploying OpenSearch (single node)"
# Auto Mode has no StorageClass out of the box; the demo's Helm values ask for one.
kubectl apply -f "${SEED_DIR}/storageclass.yaml"
if kubectl get statefulset opensearch-cluster-master -n "${NAMESPACE}" >/dev/null 2>&1; then
  echo "  Already deployed, skipping install"
else
  OPENSEARCH_STORAGE_CLASS=gp3 bash "${DEMO_DIR}/scripts/setup-opensearch.sh"
fi
kubectl rollout status statefulset/opensearch-cluster-master -n "${NAMESPACE}" --timeout=300s

# --- Knowledge base: the Bedrock (Titan) index the monolith reads ---
log "Building the Bedrock knowledge index (Titan v2, 1024d)"
BEDROCK_INDEX_NAME="${BEDROCK_INDEX_NAME:-k8s-autoscaling-knowledge-bedrock}"
OS_POD="$(kubectl get pod -n "${NAMESPACE}" -l app.kubernetes.io/name=opensearch -o jsonpath='{.items[0].metadata.name}')"
DOC_COUNT="$(kubectl exec -n "${NAMESPACE}" "${OS_POD}" -- \
  curl -s "http://localhost:9200/${BEDROCK_INDEX_NAME}/_count" 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo 0)"

if [[ "${DOC_COUNT}" -gt 100 ]]; then
  echo "  Bedrock index already populated (${DOC_COUNT} docs), skipping"
else
  # The indexer's Python deps. CodeBuild installs them in buildspec.yaml; on
  # an operator machine install into the user site only if they are missing
  # (PEP 668 distros refuse system-wide installs without the override).
  if ! python3 -c "import opensearchpy, httpx, git, requests, bs4, boto3" 2>/dev/null; then
    python3 -m pip install --quiet --user --break-system-packages \
      opensearch-py httpx gitpython requests beautifulsoup4 boto3
  fi
  kubectl port-forward -n "${NAMESPACE}" svc/opensearch-cluster-master 9200:9200 &
  PF_OS_PID=$!
  # Give the port-forward a moment to establish.
  sleep 4
  trap 'kill "${PF_OS_PID}" 2>/dev/null || true' EXIT
  AWS_REGION="${BEDROCK_REGION}" \
  BEDROCK_EMBED_MODEL="${BEDROCK_EMBED_MODEL}" \
  BEDROCK_INDEX_NAME="${BEDROCK_INDEX_NAME}" \
    python3 "${DEMO_DIR}/scripts/index-knowledge.py" --embedder=bedrock
  kill "${PF_OS_PID}" 2>/dev/null || true
  trap - EXIT
fi

# --- Demo in monolith seats ---
log "Deploying the demo (monolith seats)"
# The manifest ships CPU-first defaults and eu-west-1; patch to the monolith
# baseline and the workshop region as it is applied. sed only rewrites the
# literal placeholders and the known env defaults, leaving structure intact.
manifest="$(mktemp)"
sed -e "s|REPLACE_WITH_ECR_IMAGE|${DEMO_IMAGE}|g" \
    -e "s|REPLACE_WITH_RERANKER_IMAGE|${RERANKER_IMAGE}|g" \
    "${DEMO_DIR}/k8s-manifest.yaml" > "${manifest}"
kubectl apply -f "${manifest}"
rm -f "${manifest}"

# Set the region, Bedrock model, and monolith seats on the orchestrator. The
# analyst URL default already points at the renamed service. env patches are
# idempotent.
kubectl set env deployment/k8s-autoscaling-orchestrator -n "${NAMESPACE}" \
  AWS_REGION="${BEDROCK_REGION}" \
  AWS_DEFAULT_REGION="${BEDROCK_REGION}" \
  LLM_MODEL="${LLM_MODEL}" \
  ANALYST_URL="http://k8s-autoscaling-analyst-inference.slemify:8080" \
  TRIAGE="off" EMBED="bedrock" RERANK="off" ANALYST="llm" GATE="off"

kubectl set env deployment/k8s-autoscaling-tools -n "${NAMESPACE}" \
  AWS_REGION="${BEDROCK_REGION}" AWS_DEFAULT_REGION="${BEDROCK_REGION}"

log "Waiting for the orchestrator"
kubectl rollout status deployment/k8s-autoscaling-orchestrator -n "${NAMESPACE}" --timeout=180s

# --- Smoke query ---
log "Running a smoke query"
bash "${SEED_DIR}/smoke.sh" || {
  echo "  Smoke query failed. Check: kubectl logs -n ${NAMESPACE} deploy/k8s-autoscaling-orchestrator" >&2
  exit 1
}

log "Seed complete"
echo "The cluster is in the module 0 (monolith) start state."
echo "Open the UI:  kubectl port-forward -n ${NAMESPACE} svc/k8s-autoscaling-orchestrator 8000:80"
