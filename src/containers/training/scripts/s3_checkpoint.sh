#!/bin/bash
# S3 checkpoint sync script for Spot resilience.
# Called by training save callback to upload checkpoints to S3.
set -e

CHECKPOINT_DIR="${1:?Usage: s3_checkpoint.sh <checkpoint_dir>}"
S3_PATH="${S3_CHECKPOINT_PATH:?S3_CHECKPOINT_PATH env var required}"

echo "Syncing checkpoint to ${S3_PATH}..."
aws s3 sync "${CHECKPOINT_DIR}" "${S3_PATH}$(basename ${CHECKPOINT_DIR})/" \
    --exclude "*.bin" --exclude "optimizer*" \
    --quiet

echo "Checkpoint synced: $(basename ${CHECKPOINT_DIR})"
