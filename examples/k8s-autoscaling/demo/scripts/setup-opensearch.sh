#!/bin/bash
# Setup OpenSearch for the K8s Autoscaling Expert demo.
# Deploys a single-node OpenSearch instance in the slemify namespace.
# NOTE: Security is disabled for demo purposes. Do not use in production.
# Requires: kubectl, helm

set -e

NAMESPACE="slemify"
RELEASE="opensearch-demo"
# EKS Auto Mode clusters have no gp2 class; the workshop seed passes gp3.
STORAGE_CLASS="${OPENSEARCH_STORAGE_CLASS:-gp2}"
# Pin the chart so a new chart release does not change the OpenSearch
# version under a running workshop. Chart 3.8.0 ships OpenSearch 3.8.0.
OPENSEARCH_CHART_VERSION="${OPENSEARCH_CHART_VERSION:-3.8.0}"
# Pull from ECR Public, not Docker Hub, to avoid Docker Hub rate limits.
# Both images are multi-arch (amd64 and arm64). The busybox tag is pinned
# because the chart default for the init container is busybox:latest.
OPENSEARCH_IMAGE="${OPENSEARCH_IMAGE:-public.ecr.aws/opensearchproject/opensearch}"
OPENSEARCH_IMAGE_TAG="${OPENSEARCH_IMAGE_TAG:-3.8.0}"
INIT_IMAGE="${OPENSEARCH_INIT_IMAGE:-public.ecr.aws/docker/library/busybox}"
INIT_IMAGE_TAG="${OPENSEARCH_INIT_IMAGE_TAG:-1.37.0}"

echo "=== Setting up OpenSearch for RAG ==="

# Add OpenSearch Helm repo
echo "Adding OpenSearch Helm repo..."
helm repo add opensearch https://opensearch-project.github.io/helm-charts/ 2>/dev/null || true
helm repo update opensearch

# Deploy OpenSearch (single node, no security for demo)
echo "Deploying OpenSearch..."
helm upgrade --install $RELEASE opensearch/opensearch \
  --version "${OPENSEARCH_CHART_VERSION}" \
  --namespace $NAMESPACE \
  --set image.repository="${OPENSEARCH_IMAGE}" \
  --set image.tag="${OPENSEARCH_IMAGE_TAG}" \
  --set persistence.image="${INIT_IMAGE}" \
  --set persistence.imageTag="${INIT_IMAGE_TAG}" \
  --set replicas=1 \
  --set minimumMasterNodes=1 \
  --set persistence.size=10Gi \
  --set persistence.storageClass="${STORAGE_CLASS}" \
  --set resources.requests.memory=2Gi \
  --set resources.limits.memory=2Gi \
  --set resources.requests.cpu=1 \
  --set extraEnvs[0].name=DISABLE_SECURITY_PLUGIN \
  --set-string extraEnvs[0].value=true \
  --set config.opensearch\\.yml="plugins.security.disabled: true" \
  --set singleNode=true \
  --wait --timeout 5m

# Wait for OpenSearch to be ready
echo "Waiting for OpenSearch to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=$RELEASE \
  -n $NAMESPACE --timeout=300s

# Get the service endpoint
ENDPOINT=$(kubectl get svc -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE \
  -o jsonpath='{.items[0].metadata.name}')
echo ""
echo "=== OpenSearch ready ==="
echo "  Endpoint: http://$ENDPOINT:9200"
echo "  Namespace: $NAMESPACE"
echo ""
echo "  Test: kubectl port-forward -n $NAMESPACE svc/$ENDPOINT 9200:9200"
echo "        curl http://localhost:9200"
