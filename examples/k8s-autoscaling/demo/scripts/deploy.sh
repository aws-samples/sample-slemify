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
IMAGE="${REGISTRY}/slemify/demo-orchestrator:latest"
SCRIPT_DIR="$(dirname "$0")"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building demo orchestrator ==="
echo "  Image: ${IMAGE}"

# Login to ECR
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names slemify/demo-orchestrator \
  --region "${REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name slemify/demo-orchestrator \
  --region "${REGION}" --image-scanning-configuration scanOnPush=true

# Build and push
docker build -t "${IMAGE}" "${DEMO_DIR}"
docker push "${IMAGE}"

echo ""
echo "=== Deploying to cluster ==="

# Replace image placeholder and apply
sed "s|REPLACE_WITH_ECR_IMAGE|${IMAGE}|g" "${DEMO_DIR}/k8s-manifest.yaml" | \
  kubectl apply -f -

echo ""
echo "=== Setting up Pod Identity for Bedrock access ==="

# Check if pod identity association exists
CLUSTER_NAME=$(kubectl config current-context | grep -oP '(?<=:cluster/)[^/]+' || \
  kubectl config current-context)

ROLE_NAME="slemify-demo-orchestrator-bedrock"
SA_NAME="demo-orchestrator"
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
kubectl rollout status deployment/demo-orchestrator -n slemify --timeout=120s

echo ""
echo "=== Done ==="
echo "  Access: kubectl port-forward -n slemify svc/demo-orchestrator 8000:80"
