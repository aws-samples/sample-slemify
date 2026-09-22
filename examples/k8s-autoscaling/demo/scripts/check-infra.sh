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

ok=0; missing=0
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

echo
if [ "$missing" -eq 0 ]; then
  if [ "$ok" -eq 2 ]; then
    echo "Both ready. Module 3's 'slemify deploy --config analyst/expert.yaml' will mount the model from S3 and skip the convert."
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

if [ "$REPAIR" = "--repair" ] || [ "${REPAIR_FLAG:-}" = "1" ]; then
  echo
  echo "Starting the repair build..."
  BID="$(aws codebuild start-build --region "$REGION" --project-name "$PROVISION_PROJECT" --query 'build.id' --output text)"
  echo "  started $BID"
  echo "  follow: aws codebuild batch-get-builds --region $REGION --ids $BID --query 'builds[0].buildStatus'"
  echo "  it will re-signal nothing to Workshop Studio (the event is already ready); watch CodeBuild for SUCCEEDED."
fi
exit 1
