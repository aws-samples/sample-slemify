#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Check the two assets module 3 needs that provisioning finishes AFTER the event
# is marked ready: the Mountpoint S3 CSI addon (the analyst mounts its model
# from S3) and the analyst's converted GGUF in the model bucket. Both normally
# converge 10 to 15 minutes after the event shows ready, long before anyone
# reaches module 3. This tells you where they stand and, with --repair, re-runs
# the provisioning build, which is idempotent: Terraform converges the addon,
# the GGUF conversion skips when the file already exists, and the seed skips
# what is already there.
#
# It also checks that the workshop images are in this account's ECR (report
# only). In central mode (provisioning copied everything from the event's
# central account and wrote /slemify/mirror) it checks the latest tags and
# llama-cpp:server-b12063; otherwise it checks latest, latest-amd64 and
# latest-arm64.
#
# Usage: make check-infra            # report
#        make check-infra REPAIR=1   # report, then start the repair build
set -euo pipefail

CLUSTER="${EKS_CLUSTER_NAME:-slemify-workshop}"
BUCKET="${SLEMIFY_BUCKET:?SLEMIFY_BUCKET is not set (it is set in /etc/environment on the workshop IDE)}"
REGION="${AWS_REGION:-us-west-2}"
PROJECT="${SLEMIFY_ANALYST_PROJECT:-k8s-autoscaling-analyst}"
GGUF="${SLEMIFY_ANALYST_GGUF:-model-q4_k_m.gguf}"
PROVISION_PROJECT="${PROVISION_PROJECT:-slemify-workshop-provision}"
REPAIR="${1:-}"

# Central mode when provisioning wrote /slemify/mirror. A failed read (no
# parameter, or an older IDE role without the permission) means today's path.
MIRROR="$(aws ssm get-parameter --region "$REGION" --name /slemify/mirror --query Parameter.Value --output text 2>/dev/null || true)"
[ "$MIRROR" = "None" ] && MIRROR=""

ok=0; missing=0; checks=3
echo "Module 3 assets (finished by provisioning after the event was marked ready):"

# 1. Mountpoint S3 CSI addon
ADDON_STATUS="$(aws eks describe-addon --region "$REGION" --cluster-name "$CLUSTER" \
  --addon-name aws-mountpoint-s3-csi-driver --query 'addon.status' --output text 2>/dev/null || echo MISSING)"
case "$ADDON_STATUS" in
  ACTIVE)
    echo "  [ok]      Mountpoint S3 CSI addon: ACTIVE"; ok=$((ok+1)) ;;
  CREATING|UPDATING|DEGRADED)
    echo "  [wait]    Mountpoint S3 CSI addon: $ADDON_STATUS (still converging; its pods need a node, which the"
    echo "            reserved SLM node provides. Give it a few minutes and run this again.)" ;;
  *)
    echo "  [missing] Mountpoint S3 CSI addon: $ADDON_STATUS"; missing=$((missing+1)) ;;
esac

# 2. Analyst GGUF
if SIZE="$(aws s3api head-object --region "$REGION" --bucket "$BUCKET" --key "models/$PROJECT/$GGUF" \
     --query 'ContentLength' --output text 2>/dev/null)"; then
  echo "  [ok]      Analyst GGUF: s3://$BUCKET/models/$PROJECT/$GGUF ($((SIZE / 1073741824)) GB)"
  ok=$((ok+1))
elif [ -n "$MIRROR" ]; then
  # Central mode: provisioning copies the GGUF from the central account in its
  # tail, so the provisioning build's state tells "copying" from "failed".
  LAST="$(aws codebuild list-builds-for-project --region "$REGION" --project-name "$PROVISION_PROJECT" \
    --query 'ids[0]' --output text 2>/dev/null || echo None)"
  ST=UNKNOWN
  if [ "$LAST" != "None" ] && [ -n "$LAST" ]; then
    ST="$(aws codebuild batch-get-builds --region "$REGION" --ids "$LAST" --query 'builds[0].buildStatus' --output text 2>/dev/null || echo UNKNOWN)"
  fi
  if [ "$ST" = "IN_PROGRESS" ]; then
    echo "  [wait]    Analyst GGUF: provisioning is still copying it. About 10 minutes; run this again."
  else
    echo "  [missing] Analyst GGUF: not in the bucket, and the last provisioning build ended $ST"; missing=$((missing+1))
  fi
else
  # Distinguish "conversion still running" from "nothing running".
  RUNNING="$(aws codebuild list-builds-for-project --region "$REGION" --project-name slemify-workshop-gguf \
    --query 'ids[0]' --output text 2>/dev/null || echo None)"
  if [ "$RUNNING" != "None" ] && [ -n "$RUNNING" ]; then
    ST="$(aws codebuild batch-get-builds --region "$REGION" --ids "$RUNNING" --query 'builds[0].buildStatus' --output text 2>/dev/null || echo UNKNOWN)"
    if [ "$ST" = "IN_PROGRESS" ]; then
      echo "  [wait]    Analyst GGUF: conversion still running in CodeBuild ($RUNNING). About 14 minutes total; run this again."
    else
      echo "  [missing] Analyst GGUF: not in the bucket, and the last conversion ended $ST"; missing=$((missing+1))
    fi
  else
    echo "  [missing] Analyst GGUF: not in the bucket, no conversion running"; missing=$((missing+1))
  fi
fi

# 3. Images in this account's ECR (report only, no repair)
IMAGE_REPOS="slemify/k8s-autoscaling-orchestrator slemify/k8s-autoscaling-reranker slemify/k8s-autoscaling-decision slemify/data-pipeline slemify/classifier-trainer slemify/classifier-serving slemify/gguf-convert"
if [ -n "$MIRROR" ]; then TAGS="latest"; else TAGS="latest latest-amd64 latest-arm64"; fi
img_total=0; img_missing=0
check_tags() {  # repo tag...
  local repo="$1"; shift
  local ids=() t absent
  for t in "$@"; do ids+=("imageTag=$t"); done
  # A missing repository fails the call; count every tag as missing then.
  absent="$(aws ecr batch-get-image --region "$REGION" --repository-name "$repo" --image-ids "${ids[@]}" \
    --query "failures[?failureCode=='ImageNotFound'].imageId.imageTag" --output text 2>/dev/null)" || absent="$*"
  [ "$absent" = "None" ] && absent=""
  for t in $absent; do
    echo "  [missing] Image $repo:$t"
    img_missing=$((img_missing+1))
  done
  img_total=$((img_total + $#))
}
for repo in $IMAGE_REPOS; do
  # shellcheck disable=SC2086
  check_tags "$repo" $TAGS
done
if [ -n "$MIRROR" ]; then
  check_tags slemify/llama-cpp server-b12063
fi
if [ "$img_missing" -eq 0 ]; then
  echo "  [ok]      Images: all $img_total tags in ECR"; ok=$((ok+1))
else
  missing=$((missing+1))
fi
if [ -n "$MIRROR" ] && [ "${SLEMIFY_MIRROR_REGISTRY:-}" != "$MIRROR" ]; then
  echo "  [fix]     This terminal started before provisioning copied llama.cpp. Run: source ~/.bashrc"
fi

echo
if [ "$missing" -eq 0 ]; then
  if [ "$ok" -eq "$checks" ]; then
    echo "All ready. Module 3's 'slemify deploy --config analyst/expert.yaml' will mount the model from S3 and skip the convert."
  else
    echo "Nothing is missing, but something is still converging. This is normal in the first 15 minutes after the event shows ready."
  fi
  exit 0
fi

echo "Repair: re-run the provisioning build. It is idempotent, so it only does what is missing."
echo "  make check-infra REPAIR=1"
echo "  (or: aws codebuild start-build --region $REGION --project-name $PROVISION_PROJECT)"
echo "Without the S3 CSI addon the analyst still deploys; slemify falls back to downloading the model into the pod (slower start)."
echo "Without the GGUF, slemify deploy converts it in the cluster, which does not fit an Auto Mode node's 80 GiB disk; run the repair."
if [ "$img_missing" -ne 0 ]; then
  echo "A missing image means an image build or copy failed; the repair build retries it."
fi

if [ "$REPAIR" = "--repair" ] || [ "${REPAIR_FLAG:-}" = "1" ]; then
  echo
  echo "Starting the repair build..."
  BID="$(aws codebuild start-build --region "$REGION" --project-name "$PROVISION_PROJECT" --query 'build.id' --output text)"
  echo "  started $BID"
  echo "  follow: aws codebuild batch-get-builds --region $REGION --ids $BID --query 'builds[0].buildStatus'"
  echo "  it will re-signal nothing to Workshop Studio (the event is already ready); watch CodeBuild for SUCCEEDED."
fi
exit 1
