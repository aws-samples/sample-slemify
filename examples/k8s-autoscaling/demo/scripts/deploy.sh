#!/bin/bash
set -euo pipefail

# Deploy the demo orchestrator to the EKS cluster.
# Prerequisites:
#   - Docker running
#   - kubectl configured for the target cluster
#   - AWS credentials with ECR push access
#   - SLM models and OpenSearch already deployed

REGION="${AWS_REGION:-eu-west-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/slemify/k8s-autoscaling-orchestrator:latest"
EMBEDDING_IMAGE="${REGISTRY}/slemify/k8s-autoscaling-embedding:latest"
SCRIPT_DIR="$(dirname "$0")"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building images (multi-arch) ==="
bash "${SCRIPT_DIR}/build-images.sh"

echo ""
echo "=== Deploying to cluster ==="

# Replace image placeholders and apply
sed -e "s|REPLACE_WITH_ECR_IMAGE|${IMAGE}|g" \
    -e "s|REPLACE_WITH_EMBEDDING_IMAGE|${EMBEDDING_IMAGE}|g" \
    "${DEMO_DIR}/k8s-manifest.yaml" | \
  kubectl apply -f -

echo ""
echo "=== Setting up Pod Identity for Bedrock access ==="

# Check if pod identity association exists
CLUSTER_NAME=$(kubectl config current-context | grep -oP '(?<=:cluster/)[^/]+' || \
  kubectl config current-context)

ROLE_NAME="slemify-k8s-autoscaling-orchestrator-bedrock"
SA_NAME="k8s-autoscaling-orchestrator"
NAMESPACE="slemify"

# Create IAM role if needed
if ! aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
  echo "  Creating IAM role: ${ROLE_NAME}"
  TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "pods.eks.amazonaws.com" },
    "Action": ["sts:AssumeRole", "sts:TagSession"]
  }]
}
EOF
)
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}"

  aws iam put-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-name bedrock-access \
    --policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Action": [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ],
        "Resource": "*"
      }]
    }'
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"

# Create pod identity association if needed
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
    --role-arn "${ROLE_ARN}"
fi

echo ""
echo "=== Waiting for rollout ==="
kubectl rollout status deployment/k8s-autoscaling-orchestrator -n slemify --timeout=120s

echo ""
echo "=== Done ==="
echo "  Access: kubectl port-forward -n slemify svc/k8s-autoscaling-orchestrator 8000:80"
