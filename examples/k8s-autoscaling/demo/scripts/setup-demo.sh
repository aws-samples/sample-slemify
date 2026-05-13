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
IMAGE="${REGISTRY}/slemify/demo-orchestrator:latest"
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

# --- Step 3: Index knowledge base ---
echo "--- Step 3: Indexing knowledge base ---"

# Check if index already has data
OPENSEARCH_POD=$(kubectl get pod -n "${NAMESPACE}" -l app.kubernetes.io/name=opensearch -o jsonpath='{.items[0].metadata.name}')
DOC_COUNT=$(kubectl exec -n "${NAMESPACE}" "${OPENSEARCH_POD}" -- \
  curl -s "http://localhost:9200/k8s-autoscaling-knowledge/_count" 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo "0")

if [ "${DOC_COUNT}" -gt 100 ]; then
  echo "  Knowledge base already indexed (${DOC_COUNT} docs), skipping"
else
  echo "  Installing Python dependencies..."
  pip install --quiet opensearch-py boto3 gitpython requests beautifulsoup4

  echo "  Starting port-forward for indexing..."
  kubectl port-forward -n "${NAMESPACE}" svc/opensearch-cluster-master 9200:9200 &
  PF_PID=$!
  sleep 3

  python3 "${SCRIPT_DIR}/index-knowledge.py"

  kill "${PF_PID}" 2>/dev/null
fi
echo ""

# --- Step 4: Build and push orchestrator image ---
echo "--- Step 4: Building orchestrator image (arm64) ---"

# ECR login
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}" 2>/dev/null

# Create ECR repo if needed
aws ecr describe-repositories --repository-names slemify/demo-orchestrator \
  --region "${REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name slemify/demo-orchestrator \
  --region "${REGION}" --image-scanning-configuration scanOnPush=true >/dev/null

# Build natively on arm64
if [ -n "${ARM64_BUILD_HOST:-}" ]; then
  echo "  Building on remote host: ${ARM64_BUILD_HOST}"
  ssh "${ARM64_BUILD_HOST}" "mkdir -p ~/demo-build"
  scp "${DEMO_DIR}/Dockerfile" "${DEMO_DIR}/server.py" "${ARM64_BUILD_HOST}:~/demo-build/"
  ssh "${ARM64_BUILD_HOST}" "aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REGISTRY} 2>/dev/null && docker build -t ${IMAGE} ~/demo-build/ && docker push ${IMAGE}"
elif [ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; then
  echo "  Building locally (native arm64)"
  docker build -t "${IMAGE}" "${DEMO_DIR}"
  docker push "${IMAGE}"
else
  echo "  ERROR: Native arm64 build required but running on $(uname -m)"
  echo "  Set ARM64_BUILD_HOST to an SSH-accessible Graviton instance:"
  echo "    ARM64_BUILD_HOST=my-graviton-host ./setup-demo.sh"
  exit 1
fi
echo "  Image pushed: ${IMAGE}"
echo ""

# --- Step 5: Deploy orchestrator ---
echo "--- Step 5: Deploying orchestrator ---"

sed "s|REPLACE_WITH_ECR_IMAGE|${IMAGE}|g" "${DEMO_DIR}/k8s-manifest.yaml" | \
  kubectl apply -f -

# --- Step 6: Pod Identity for Bedrock ---
echo ""
echo "--- Step 6: Setting up Pod Identity ---"

CLUSTER_NAME=$(aws eks list-clusters --region "${REGION}" --query 'clusters[0]' --output text)
ROLE_NAME="slemify-demo-orchestrator-bedrock"
SA_NAME="demo-orchestrator"

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
kubectl rollout status deployment/demo-orchestrator -n "${NAMESPACE}" --timeout=120s

echo ""
echo "=== Demo setup complete ==="
echo ""
echo "To run the demo:"
echo "  ./scripts/demo-terminal.sh"
echo ""
echo "Or manually:"
echo "  kubectl port-forward -n slemify svc/demo-orchestrator 8000:80"
echo "  open http://localhost:8000"
