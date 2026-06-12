#!/bin/bash
set -euo pipefail

# Full demo setup: deploys OpenSearch, indexes knowledge base, builds and
# deploys the orchestrator. Run once before the live demo.
#
# Prerequisites:
#   - kubectl configured for the target EKS cluster
#   - AWS credentials with ECR, Bedrock, and EKS access
#   - SLM models already deployed (slemify deploy)
#   - Python 3.12+ with pip
#   - Docker available (native arm64 build required)
#     Set ARM64_BUILD_HOST=<ssh-host> if building remotely on a Graviton instance
#
# Usage:
#   ./setup-demo.sh

SCRIPT_DIR="$(dirname "$0")"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
REGION="${AWS_REGION:-eu-west-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/slemify/k8s-autoscaling-orchestrator:latest"
EMBEDDING_IMAGE="${REGISTRY}/slemify/k8s-autoscaling-embedding:latest"
RERANKER_IMAGE="${REGISTRY}/slemify/k8s-autoscaling-reranker:latest"
NAMESPACE="slemify"

echo "=== Demo Setup ==="
echo "  Region: ${REGION}"
echo "  Account: ${ACCOUNT}"
echo "  Image: ${IMAGE}"
echo ""

# --- Step 1: Verify prerequisites ---
echo "--- Step 1: Verifying prerequisites ---"

# Check SLM models are deployed
if ! kubectl get deployment -n "${NAMESPACE}" -l slemify.io/stage=serving -o name | grep -q .; then
  echo "  ERROR: No SLM inference deployments found in namespace ${NAMESPACE}"
  echo "  Run 'slemify deploy' for your models first."
  exit 1
fi
echo "  SLM models: deployed"

# Check kubectl access
if ! kubectl get ns "${NAMESPACE}" >/dev/null 2>&1; then
  echo "  ERROR: Cannot access namespace ${NAMESPACE}"
  exit 1
fi
echo "  Cluster access: ok"
echo ""

# --- Step 2: Deploy OpenSearch ---
echo "--- Step 2: Deploying OpenSearch ---"

if kubectl get statefulset opensearch-cluster-master -n "${NAMESPACE}" >/dev/null 2>&1; then
  echo "  OpenSearch already deployed, skipping"
else
  bash "${SCRIPT_DIR}/setup-opensearch.sh"
fi

# Wait for OpenSearch to be ready
echo "  Waiting for OpenSearch to be ready..."
kubectl rollout status statefulset/opensearch-cluster-master -n "${NAMESPACE}" --timeout=180s
echo "  OpenSearch: ready"
echo ""

# --- Step 3: Build and push images (multi-arch) ---
echo "--- Step 3: Building images (multi-arch) ---"
bash "${SCRIPT_DIR}/build-images.sh"
echo "  Images pushed"
echo ""

# --- Step 4: Deploy manifests (embedding + reranker + orchestrator) ---
echo "--- Step 4: Deploying manifests ---"

sed -e "s|REPLACE_WITH_ECR_IMAGE|${IMAGE}|g" \
    -e "s|REPLACE_WITH_EMBEDDING_IMAGE|${EMBEDDING_IMAGE}|g" \
    -e "s|REPLACE_WITH_RERANKER_IMAGE|${RERANKER_IMAGE}|g" \
    "${DEMO_DIR}/k8s-manifest.yaml" | \
  kubectl apply -f -

echo "  Waiting for embedding and reranker services to be ready..."
kubectl rollout status deployment/k8s-autoscaling-embedding -n "${NAMESPACE}" --timeout=300s
kubectl rollout status deployment/k8s-autoscaling-reranker -n "${NAMESPACE}" --timeout=300s
echo "  Embedding + reranker services: ready"
echo ""

# --- Step 5: Index knowledge base ---
echo "--- Step 5: Indexing knowledge base ---"

# Knowledge base index, built with the in-cluster bge-base embeddings (768d).
INDEX_NAME="k8s-autoscaling-knowledge"

# Check if index already has data
OPENSEARCH_POD=$(kubectl get pod -n "${NAMESPACE}" -l app.kubernetes.io/name=opensearch -o jsonpath='{.items[0].metadata.name}')
DOC_COUNT=$(kubectl exec -n "${NAMESPACE}" "${OPENSEARCH_POD}" -- \
  curl -s "http://localhost:9200/${INDEX_NAME}/_count" 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo "0")

if [ "${DOC_COUNT}" -gt 100 ]; then
  echo "  Knowledge base already indexed (${DOC_COUNT} docs), skipping"
else
  echo "  Installing Python dependencies..."
  pip install --quiet opensearch-py httpx gitpython requests beautifulsoup4

  echo "  Starting port-forwards for indexing..."
  kubectl port-forward -n "${NAMESPACE}" svc/opensearch-cluster-master 9200:9200 &
  PF_OS_PID=$!
  kubectl port-forward -n "${NAMESPACE}" svc/k8s-autoscaling-embedding 8083:80 &
  PF_EMB_PID=$!
  sleep 3

  python3 "${SCRIPT_DIR}/index-knowledge.py" --index-name="${INDEX_NAME}"

  kill "${PF_OS_PID}" "${PF_EMB_PID}" 2>/dev/null
fi
echo ""

# --- Step 6: Pod Identity for Bedrock ---
echo ""
echo "--- Step 6: Setting up Pod Identity ---"

CLUSTER_NAME=$(aws eks list-clusters --region "${REGION}" --query 'clusters[0]' --output text)
ROLE_NAME="slemify-k8s-autoscaling-orchestrator-bedrock"
SA_NAME="k8s-autoscaling-orchestrator"

if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Creating IAM role: ${ROLE_NAME}"
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": { "Service": "pods.eks.amazonaws.com" },
        "Action": ["sts:AssumeRole", "sts:TagSession"]
      }]
    }' >/dev/null

  aws iam put-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-name bedrock-access \
    --policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
        "Resource": "*"
      }]
    }'
  echo "  Role created"
else
  echo "  IAM role already exists"
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"

EXISTING=$(aws eks list-pod-identity-associations \
  --cluster-name "${CLUSTER_NAME}" \
  --namespace "${NAMESPACE}" \
  --service-account "${SA_NAME}" \
  --query 'associations[0].associationId' --output text 2>/dev/null || echo "None")

if [ "${EXISTING}" = "None" ] || [ -z "${EXISTING}" ]; then
  echo "  Creating Pod Identity association"
  aws eks create-pod-identity-association \
    --cluster-name "${CLUSTER_NAME}" \
    --namespace "${NAMESPACE}" \
    --service-account "${SA_NAME}" \
    --role-arn "${ROLE_ARN}" >/dev/null
else
  echo "  Pod Identity association already exists"
fi

# --- Step 7: Wait for rollout ---
echo ""
echo "--- Step 7: Waiting for orchestrator to be ready ---"
kubectl rollout status deployment/k8s-autoscaling-orchestrator -n "${NAMESPACE}" --timeout=120s

echo ""
echo "=== Demo setup complete ==="
echo ""
echo "To run the demo:"
echo "  ./scripts/demo-terminal.sh"
echo ""
echo "Or manually:"
echo "  kubectl port-forward -n slemify svc/k8s-autoscaling-orchestrator 8000:80"
echo "  open http://localhost:8000"
