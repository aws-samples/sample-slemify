#!/bin/bash
set -euo pipefail

# Build the demo orchestrator and reranker images as multi-arch
# (linux/amd64 + linux/arm64) and publish manifest lists to ECR.
#
# Builds run natively on two remote hosts (no QEMU emulation), matching the
# documented slemify build pattern:
#   ARM64_BUILD_HOST  Graviton/arm64 host (required)
#   X86_BUILD_HOST    x86_64 host (default: x86-host from your SSH config)
#
# Both hosts need docker (with buildx) and AWS credentials for ECR.

REGION="${AWS_REGION:-eu-west-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
ARM64_BUILD_HOST="${ARM64_BUILD_HOST:-}"
X86_BUILD_HOST="${X86_BUILD_HOST:-x86-host}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

ORCH_IMAGE="${REGISTRY}/slemify/k8s-autoscaling-orchestrator:latest"
RERANK_IMAGE="${REGISTRY}/slemify/k8s-autoscaling-reranker:latest"

if [ -z "${ARM64_BUILD_HOST}" ]; then
  echo "ERROR: ARM64_BUILD_HOST must be set (e.g. ARM64_BUILD_HOST=my-graviton-host)"
  exit 1
fi

echo "=== Ensuring ECR repositories ==="
for repo in slemify/k8s-autoscaling-orchestrator slemify/k8s-autoscaling-reranker; do
  aws ecr describe-repositories --repository-names "${repo}" --region "${REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${repo}" --region "${REGION}" \
      --image-scanning-configuration scanOnPush=true >/dev/null
done

# docker_prefix <host>
# Some build hosts run docker as the login user; others require sudo. Detect
# which so login/build/push all use the same docker config (credentials must
# match the daemon connection).
docker_prefix() {
  local host="$1"
  if ssh "${host}" "docker ps >/dev/null 2>&1"; then
    echo ""
  elif ssh "${host}" "sudo -n docker ps >/dev/null 2>&1"; then
    echo "sudo"
  else
    echo "ERROR: docker not usable on ${host} (with or without passwordless sudo)" >&2
    exit 1
  fi
}

# sync_and_build <host> <arch>
# Syncs the demo dir to the host and builds both images natively for that arch.
sync_and_build() {
  local host="$1" arch="$2"
  local dkr
  dkr="$(docker_prefix "${host}")"
  echo "=== Building ${arch} natively on ${host} (docker: ${dkr:-direct}) ==="
  rsync -az --delete --exclude='__pycache__' --exclude='*.pyc' \
    "${DEMO_DIR}/" "${host}:~/demo-build/"
  ssh "${host}" "aws ecr get-login-password --region ${REGION} | ${dkr} docker login --username AWS --password-stdin ${REGISTRY}"
  ssh "${host}" "cd ~/demo-build && ${dkr} docker build -t ${ORCH_IMAGE}-${arch} . && ${dkr} docker push ${ORCH_IMAGE}-${arch}"
  ssh "${host}" "cd ~/demo-build/reranker && ${dkr} docker build -t ${RERANK_IMAGE}-${arch} . && ${dkr} docker push ${RERANK_IMAGE}-${arch}"
}

sync_and_build "${ARM64_BUILD_HOST}" arm64
sync_and_build "${X86_BUILD_HOST}" amd64

# Combine the per-arch images into a single multi-arch manifest list. imagetools
# reads the already-pushed images from the registry (no rebuild). Run it on the
# arm64 host, which is already logged in to ECR.
echo "=== Creating multi-arch manifests ==="
ARM_DKR="$(docker_prefix "${ARM64_BUILD_HOST}")"
ssh "${ARM64_BUILD_HOST}" "${ARM_DKR} docker buildx imagetools create -t ${ORCH_IMAGE} ${ORCH_IMAGE}-amd64 ${ORCH_IMAGE}-arm64"
ssh "${ARM64_BUILD_HOST}" "${ARM_DKR} docker buildx imagetools create -t ${RERANK_IMAGE} ${RERANK_IMAGE}-amd64 ${RERANK_IMAGE}-arm64"

echo "=== Multi-arch images published ==="
echo "  ${ORCH_IMAGE} (amd64 + arm64)"
echo "  ${RERANK_IMAGE} (amd64 + arm64)"
