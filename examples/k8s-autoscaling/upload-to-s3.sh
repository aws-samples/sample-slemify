#!/bin/bash
# Upload query files to S3 for the Slemify data pipeline.
# Creates the bucket if it doesn't exist.
#
# Usage:
#   ./upload-to-s3.sh [bucket-name]
#
# Default bucket: slemify-data

set -euo pipefail

DIR="$(dirname "$0")/data/queries"
PREFIX="k8s-autoscaling/data/queries"
REGION="${AWS_DEFAULT_REGION:-eu-west-1}"
BUCKET="${1:-slemify-data}"

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR not found"
    exit 1
fi

# Create bucket if it doesn't exist
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    echo "Creating bucket s3://$BUCKET in $REGION..."
    aws s3api create-bucket \
        --bucket "$BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
    echo "Bucket created."
fi

COUNT=$(ls "$DIR"/*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "Uploading $COUNT queries to s3://$BUCKET/$PREFIX/"

aws s3 sync "$DIR" "s3://$BUCKET/$PREFIX/" --exclude "*" --include "*.txt"

# Upload eval queries if they exist
EVAL_DIR="$(dirname "$0")/data/eval-queries"
EVAL_PREFIX="k8s-autoscaling/data/eval-queries"
if [ -d "$EVAL_DIR" ]; then
    EVAL_COUNT=$(ls "$EVAL_DIR"/*.txt 2>/dev/null | wc -l | tr -d ' ')
    echo "Uploading $EVAL_COUNT eval queries to s3://$BUCKET/$EVAL_PREFIX/"
    aws s3 sync "$EVAL_DIR" "s3://$BUCKET/$EVAL_PREFIX/" --exclude "*" --include "*.txt"
fi

echo ""
echo "Done. Bucket: $BUCKET"
echo "Use this in your expert.yaml:"
echo "  data:"
echo "    bucket: $BUCKET"
echo "    path: k8s-autoscaling/data/"
