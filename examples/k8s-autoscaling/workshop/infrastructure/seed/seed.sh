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
# Failure injection for test events (off unless FAULT_INJECTION is set).
# shellcheck source=../lib/fault.sh
. "${INFRA_DIR}/lib/fault.sh"

# Each phase header carries the time since the seed started, and the phases
# are summed up at the end, so the CodeBuild log shows where the time goes.
SEED_PHASES=""
log() {
  printf '\n=== [+%02dm%02ds] %s ===\n' $((SECONDS / 60)) $((SECONDS % 60)) "$1"
  SEED_PHASES="${SEED_PHASES}${SECONDS}	$1
"
}
phase_timings() {
  printf '\n=== Seed phase timings ===\n'
  printf '%s' "$SEED_PHASES" | awk -F '\t' -v end="$SECONDS" '
    { t[NR] = $1; m[NR] = $2 }
    END {
      for (i = 1; i <= NR; i++) {
        d = ((i < NR) ? t[i + 1] : end) - t[i]
        printf "  %3dm%02ds  %s\n", int(d / 60), d % 60, m[i]
      }
      printf "  total %dm%02ds\n", int(end / 60), end % 60
    }'
  # Background stages overlap the phases above, so their time is not in the
  # phase rows. Each one is recorded when the seed joins it.
  if [[ -n "$SEED_BG" ]]; then
    printf '  background, overlapped with the phases above:\n'
    printf '%s' "$SEED_BG" | awk -F '\t' '{ printf "  %3dm%02ds  %s\n", int($1 / 60), $1 % 60, $2 }'
  fi
}
SEED_BG=""
PF_OS_PID=""
# The summary prints from the EXIT trap, so a failed seed still shows where
# the time went, and the failing phase is the last row.
on_exit() {
  local rc=$?
  if [[ -n "$PF_OS_PID" ]]; then
    kill "$PF_OS_PID" 2>/dev/null || true
  fi
  fault_bedrock_stop "${MODEL_BUCKET:-}" || true
  if [[ -n "$SEED_PHASES" ]]; then
    phase_timings || true
    if [[ "$rc" -ne 0 ]]; then
      printf '  seed failed (exit %d) in the last phase above\n' "$rc"
    fi
  fi
  exit "$rc"
}
trap on_exit EXIT

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

# Bedrock model. re:Invent 2026 requires Claude Sonnet 5 (4.5/4.6 are on the
# do-not-use list) via the global Cross-Region Inference profile, which routes
# from the vended account's region (Bedrock is in us-east-1/us-west-2 there).
LLM_MODEL="${LLM_MODEL:-global.anthropic.claude-sonnet-5}"
BEDROCK_EMBED_MODEL="${BEDROCK_EMBED_MODEL:-amazon.titan-embed-text-v2:0}"

# Demo images. Default to the account's ECR (built by build-images.sh); override
# for a shared registry.
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY_DEFAULT="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
DEMO_IMAGE="${DEMO_IMAGE:-${REGISTRY_DEFAULT}/slemify/k8s-autoscaling-orchestrator:latest}"
RERANKER_IMAGE="${RERANKER_IMAGE:-${REGISTRY_DEFAULT}/slemify/k8s-autoscaling-reranker:latest}"
DECISION_IMAGE="${DECISION_IMAGE:-${REGISTRY_DEFAULT}/slemify/k8s-autoscaling-decision:latest}"

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

# --- Slemify config, and the example data both the index and the projects need ---
# Both projects run only their data stage here (the frontier model writing the
# training examples): module 1 has attendees train the triage classifier, and
# module 2 has them fine-tune the retriever, so the seed leaves both at data
# done with training/serving for the attendee. Slemify reads the bucket and
# model from these two variables instead of the shipped expert.yaml values.
EXAMPLE_DIR="$(cd "${INFRA_DIR}/../.." && pwd)"
export SLEMIFY_BUCKET="${MODEL_BUCKET}"
export SLEMIFY_BEDROCK_MODEL="${LLM_MODEL}"
export AWS_REGION="${REGION}"

log "Uploading the example data to s3://${MODEL_BUCKET}/k8s-autoscaling/data/"
AWS_DEFAULT_REGION="${REGION}" bash "${EXAMPLE_DIR}/upload-to-s3.sh" "${MODEL_BUCKET}"

# Both data stages are Bedrock-only Jobs (the frontier model writing training
# examples) with no dependency on the Titan index below, so start them now and
# let them run while the index builds. They are waited on before the seed ends.
# Each one writes its end time (epoch seconds) to <log>.end, so the join can
# report the stage's own wall time even when it finished long before.
TRIAGE_DATA_PID=""
RETRIEVER_DATA_PID=""
# bedrock-unavailable (lib/fault.sh): the data pipeline reads this window at start.
fault_bedrock_start "${MODEL_BUCKET}"
start_data() {  # config logfile; sets DATA_PID and DATA_T0
  rm -f "$2.end"
  DATA_T0="$(date +%s)"
  (
    cd "${EXAMPLE_DIR}" || exit 1
    rc=0
    slemify deploy --config "$1" --until data || rc=$?
    date +%s > "$2.end"
    exit "$rc"
  ) > "$2" 2>&1 &
  DATA_PID=$!
}
if command -v slemify >/dev/null 2>&1; then
  log "Triage: starting the data stage in the background (Bedrock writes the training examples)"
  start_data triage/expert.yaml /tmp/triage-data.log
  TRIAGE_DATA_PID="${DATA_PID}"
  TRIAGE_DATA_T0="${DATA_T0}"
  log "Retriever: starting the data stage in the background (Bedrock writes the (query, chunk) pairs)"
  start_data retriever/expert.yaml /tmp/retriever-data.log
  RETRIEVER_DATA_PID="${DATA_PID}"
  RETRIEVER_DATA_T0="${DATA_T0}"
fi

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
  # Give the port-forward a moment to establish. on_exit kills it if the
  # indexer fails.
  sleep 4
  # In central mode the build copied the prep corpus snapshot into the model
  # bucket, so this index and the module 2 tuned index hold the same chunks.
  INDEX_ARGS=(--embedder=bedrock)
  if [ -n "${KNOWLEDGE_CORPUS:-}" ]; then
    INDEX_ARGS+=(--from-corpus="${KNOWLEDGE_CORPUS}")
  fi
  BEDROCK_REGION="${BEDROCK_REGION}" \
  BEDROCK_EMBED_MODEL="${BEDROCK_EMBED_MODEL}" \
  BEDROCK_INDEX_NAME="${BEDROCK_INDEX_NAME}" \
    python3 "${DEMO_DIR}/scripts/index-knowledge.py" "${INDEX_ARGS[@]}"
  kill "${PF_OS_PID}" 2>/dev/null || true
  PF_OS_PID=""
fi

# --- Slemify data stages: join the two background jobs ---
# Both leave their project at "data done", which is where the module pages
# start: module 1 has attendees train the triage classifier, module 2 has them
# fine-tune the retriever. The tuned knowledge index is not built here; it has
# to be built against the encoder the attendee trains (see `make index-tuned`).
if command -v slemify >/dev/null 2>&1; then
  wait_data() {  # name pid logfile start_epoch project
    [[ -z "$2" ]] && return 0
    log "$1: waiting for the background data stage"
    local rc=0 end took
    wait "$2" || rc=$?
    # The stage's own end time; if it is missing, it ended no later than now.
    end="$(cat "$3.end" 2>/dev/null || date +%s)"
    took=$((end - $4))
    printf '  %s data stage: %dm%02ds wall time in the background (exit %d)\n' \
      "$1" $((took / 60)) $((took % 60)) "$rc"
    SEED_BG="${SEED_BG}${took}	$1 data stage (exit ${rc})
"
    if [[ "$rc" -eq 0 ]]; then
      tail -n 4 "$3" || true
    else
      echo "ERROR: $1 data stage failed" >&2
      cat "$3" >&2
      # The Job controller deletes the failed pods, so their logs are gone.
      # The pipeline leaves the cause in the bucket when it fails on purpose.
      { aws s3 cp "s3://${MODEL_BUCKET}/$5/processed/data-failure.json" - 2>/dev/null || true; } >&2
      exit 1
    fi
  }
  wait_data "Triage" "${TRIAGE_DATA_PID}" /tmp/triage-data.log "${TRIAGE_DATA_T0:-0}" k8s-autoscaling-triage
  wait_data "Retriever" "${RETRIEVER_DATA_PID}" /tmp/retriever-data.log "${RETRIEVER_DATA_T0:-0}" k8s-autoscaling-retriever
  fault_bedrock_stop "${MODEL_BUCKET}"
else
  echo "  slemify not installed; skipping the triage and retriever data stages" >&2
fi

# --- Demo in monolith seats ---
log "Deploying the demo (monolith seats)"
# The manifest ships CPU-first defaults and eu-west-1; patch to the monolith
# baseline and the workshop region as it is applied. sed only rewrites the
# literal placeholders and the known env defaults, leaving structure intact.
manifest="$(mktemp)"
sed -e "s|REPLACE_WITH_ECR_IMAGE|${DEMO_IMAGE}|g" \
    -e "s|REPLACE_WITH_RERANKER_IMAGE|${RERANKER_IMAGE}|g" \
    -e "s|REPLACE_WITH_DECISION_IMAGE|${DECISION_IMAGE}|g" \
    "${DEMO_DIR}/k8s-manifest.yaml" > "${manifest}"
# The manifest ships the re-ranker (10 CPU) and decision model (4 CPU) at one
# replica, and the demo scripts rely on that. Applied as is, both pods preempt
# the 30 CPU slm-prewarm pod, and the replacement prewarm pod then makes Auto
# Mode launch a second SLM node that consolidation removes minutes later. Set
# both Deployments to 0 in the applied copy so they never start. Only the two
# Deployment documents change (the Services carry the same names).
python3 - "${manifest}" k8s-autoscaling-reranker k8s-autoscaling-decision <<'PY'
import re
import sys
path, names = sys.argv[1], sys.argv[2:]
with open(path) as f:
    docs = re.split(r"(?m)^(?=---[ \t]*$)", f.read())
for name in names:
    hits = [i for i, d in enumerate(docs)
            if re.search(r"(?m)^kind: Deployment[ \t]*$", d)
            and re.search(r"(?m)^  name: %s[ \t]*$" % re.escape(name), d)]
    if len(hits) != 1:
        sys.exit("expected one Deployment %s, found %d" % (name, len(hits)))
    docs[hits[0]], n = re.subn(r"(?m)^  replicas: \d+[ \t]*$", "  replicas: 0",
                               docs[hits[0]], count=1)
    if n != 1:
        sys.exit("no spec.replicas in Deployment %s" % name)
with open(path, "w") as f:
    f.write("".join(docs))
PY
kubectl apply -f "${manifest}"
rm -f "${manifest}"

# The re-ranker arrives in module 2, where the attendee scales it up from zero.
# It is already at zero in the applied manifest; the scale calls guard against
# the rewrite above missing. This matches the module's "a new model comes
# online" narrative and keeps the starting state minimal (orchestrator, tools,
# OpenSearch, and the prewarm pause pod).
kubectl scale deployment/k8s-autoscaling-reranker -n "${NAMESPACE}" --replicas=0
# The decision model is module 4's optional gate; same pattern, scaled up there.
kubectl scale deployment/k8s-autoscaling-decision -n "${NAMESPACE}" --replicas=0

# Set the region, Bedrock model, and monolith seats on the orchestrator. The
# analyst URL default already points at the renamed service. env patches are
# idempotent.
kubectl set env deployment/k8s-autoscaling-orchestrator -n "${NAMESPACE}" \
  AWS_REGION="${BEDROCK_REGION}" \
  AWS_DEFAULT_REGION="${BEDROCK_REGION}" \
  BEDROCK_REGION="${BEDROCK_REGION}" \
  LLM_MODEL="${LLM_MODEL}" \
  ANALYST_URL="http://k8s-autoscaling-analyst-inference.slemify:8080" \
  TRIAGE="off" EMBED="bedrock" RERANK="off" ANALYST="llm" GATE="off"

kubectl set env deployment/k8s-autoscaling-tools -n "${NAMESPACE}" \
  AWS_REGION="${BEDROCK_REGION}" AWS_DEFAULT_REGION="${BEDROCK_REGION}" \
  BEDROCK_REGION="${BEDROCK_REGION}"

log "Waiting for the orchestrator"
kubectl rollout status deployment/k8s-autoscaling-orchestrator -n "${NAMESPACE}" --timeout=180s

# --- Smoke query ---
log "Running a smoke query"
bash "${SEED_DIR}/smoke.sh" || {
  echo "  Smoke query failed. Check: kubectl logs -n ${NAMESPACE} deploy/k8s-autoscaling-orchestrator" >&2
  exit 1
}

# on_exit prints the phase timings after these lines.
log "Seed complete"
echo "The cluster is in the module 0 (monolith) start state."
echo "Open the UI:  kubectl port-forward -n ${NAMESPACE} svc/k8s-autoscaling-orchestrator 8000:80"
