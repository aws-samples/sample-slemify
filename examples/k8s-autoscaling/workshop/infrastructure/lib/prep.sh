#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Central prep build driver, run by buildspec-prep.yaml in the central
# account. Builds once what every team copies: the seven images (by starting
# the two image projects), the knowledge corpus, the llama.cpp mirror, and the
# analyst GGUF (by starting the GGUF project). Everything goes under
# releases/<tag>/<commit>/ in MODEL_BUCKET; images-ready.json is written once
# the images and corpus are in place, and ready.json is written last.
#
# A rerun at the same commit skips what exists. FORCE=1 redoes every step.
#
# Environment (set by the central stack):
#   RELEASE_TAG                                tag the build source was cloned at
#   MODEL_BUCKET                               central artifact bucket
#   IMAGES_AMD64_PROJECT, IMAGES_ARM64_PROJECT image builds (buildspec-images.yaml)
#   GGUF_PROJECT                               GGUF conversion (buildspec-gguf.yaml)
#   FORCE                                      1 to ignore every skip
#   CRANE_VERSION, CRANE_SHA256                crane release and its sha256 (buildspec-prep.yaml)
#   CODEBUILD_RESOLVED_SOURCE_VERSION, CODEBUILD_SRC_DIR, CODEBUILD_BUILD_ID, AWS_REGION

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
. "$HERE/central.sh"

FORCE="${FORCE:-0}"
GGUF_PID=""

step() {
  echo "$1" > "$CENTRAL_TMP/prep-step"
  echo "=== prep: $1 ==="
}

skip_allowed() { [ "$FORCE" != 1 ]; }

# have_tags <arch>: 0 when all seven <commit12>-<arch> tags exist.
have_tags() {
  local repo
  for repo in $CENTRAL_BUILT_REPOS; do
    aws ecr describe-images --repository-name "$repo" --image-ids "imageTag=${C12}-$1" >/dev/null 2>&1 || return 1
  done
}

# start_images <project> <arch>: prints the build id, or nothing when skipped.
start_images() {
  if skip_allowed && have_tags "$2"; then
    echo "all seven ${C12}-$2 tags exist; not starting $1" >&2
    return 0
  fi
  retry 3 0 aws codebuild start-build --project-name "$1" --source-version "$COMMIT" \
    --environment-variables-override "name=IMAGE_TAG,value=${C12},type=PLAINTEXT" \
    --query build.id --output text
}

# wait_build <id> <label>: polls every 20 s; 0 on SUCCEEDED, 1 otherwise.
wait_build() {
  [ -n "$1" ] || return 0
  local status unknown=0
  while :; do
    status=$(aws codebuild batch-get-builds --ids "$1" --query 'builds[0].buildStatus' --output text 2>/dev/null) || status=UNKNOWN
    echo "  $(date +%H:%M:%S) $2 ${status}"
    case "$status" in
      SUCCEEDED) return 0 ;;
      IN_PROGRESS) unknown=0 ;;
      UNKNOWN)
        unknown=$((unknown + 1))
        if [ "$unknown" -ge 10 ]; then
          echo "$2 build status could not be read 10 times in a row"
          return 1
        fi ;;
      *) echo "$2 build ended ${status}"; return 1 ;;
    esac
    sleep 20
  done
}

# gguf_in_progress: 0 when a recent build of GGUF_PROJECT is IN_PROGRESS.
gguf_in_progress() {
  local ids statuses
  ids=$(aws codebuild list-builds-for-project --project-name "$GGUF_PROJECT" --query 'ids[0:5]' --output text 2>/dev/null) || return 1
  [ -n "$ids" ] && [ "$ids" != None ] || return 1
  # shellcheck disable=SC2086
  statuses=$(aws codebuild batch-get-builds --ids $ids --query 'builds[].buildStatus' --output text 2>/dev/null) || return 1
  case "$statuses" in *IN_PROGRESS*) return 0 ;; esac
  return 1
}

# gguf_track: one GGUF build at a time, then a size check on the key.
gguf_track() {
  local key="${PREFIX}${CENTRAL_GGUF_KEY}" id size
  if skip_allowed && s3_exists "$MODEL_BUCKET" "$key"; then
    echo "GGUF exists; skipping the conversion"
    return 0
  fi
  while gguf_in_progress; do
    echo "another ${GGUF_PROJECT} build is running; waiting for it"
    sleep 30
  done
  if skip_allowed && s3_exists "$MODEL_BUCKET" "$key"; then
    echo "GGUF appeared while waiting; skipping the conversion"
    return 0
  fi
  id=$(retry 3 0 aws codebuild start-build --project-name "$GGUF_PROJECT" --source-version "$COMMIT" \
    --environment-variables-override \
      "name=MODEL_BUCKET,value=${MODEL_BUCKET},type=PLAINTEXT" \
      "name=GGUF_IMAGE_TAG,value=${C12}-amd64,type=PLAINTEXT" \
      "name=KEY_PREFIX,value=${PREFIX},type=PLAINTEXT" \
    --query build.id --output text) || return 1
  echo "GGUF build started: ${id}"
  wait_build "$id" gguf || return 1
  size=$(s3_size "$MODEL_BUCKET" "$key") || return 1
  if ! [ "$size" -gt 0 ] 2>/dev/null; then
    echo "GGUF key ${key} is empty or missing"
    return 1
  fi
  echo "GGUF ready (${size} bytes)"
}

stop_gguf_track() {
  if [ -n "$GGUF_PID" ]; then
    kill "$GGUF_PID" 2>/dev/null || true
  fi
}
trap stop_gguf_track EXIT

step validate
tag_re='^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$'
commit_re='^[0-9a-f]{40}$'
if ! [[ "${RELEASE_TAG:-}" =~ $tag_re ]]; then
  echo "RELEASE_TAG is not a valid tag (letters, digits, . _ -, at most 128)"
  exit 1
fi
COMMIT="${CODEBUILD_RESOLVED_SOURCE_VERSION:-}"
if ! [[ "$COMMIT" =~ $commit_re ]]; then
  echo "CODEBUILD_RESOLVED_SOURCE_VERSION is not a 40-character commit"
  exit 1
fi
C12="${COMMIT:0:12}"
PREFIX="releases/${RELEASE_TAG}/${COMMIT}/"
echo "$PREFIX" > "$CENTRAL_TMP/prep-prefix"
ACCOUNT=$(retry 3 0 aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo "preparing ${RELEASE_TAG} at ${COMMIT} into s3://${MODEL_BUCKET}/${PREFIX}"
# A failed.json from an earlier run would mislead waiting teams. Non-fatal.
aws s3 rm --quiet "s3://${MODEL_BUCKET}/${PREFIX}failed.json" >/dev/null 2>&1 || true

if skip_allowed && s3_exists "$MODEL_BUCKET" "${PREFIX}ready.json"; then
  echo "already prepared: ${PREFIX}ready.json exists (FORCE=1 to rebuild)"
  exit 0
fi

# Tool setup runs here, after the prefix is known, so a failure still gets a
# failed.json from post_build. CRANE_VERSION and CRANE_SHA256 come from
# buildspec-prep.yaml.
step setup
curl -fsSL --retry 5 --retry-all-errors --retry-delay 5 -o "$CENTRAL_TMP/crane.tgz" \
  "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz"
echo "${CRANE_SHA256}  $CENTRAL_TMP/crane.tgz" | sha256sum -c -
tar -xzf "$CENTRAL_TMP/crane.tgz" -C /usr/local/bin crane
rm -f "$CENTRAL_TMP/crane.tgz"
# The corpus export imports these at the top level.
pip3 install --quiet --retries 10 --timeout 60 opensearch-py httpx gitpython requests beautifulsoup4 boto3

step start-images
AMD_ID=$(start_images "$IMAGES_AMD64_PROJECT" amd64)
ARM_ID=$(start_images "$IMAGES_ARM64_PROJECT" arm64)
echo "image builds: amd64 ${AMD_ID:-skipped}, arm64 ${ARM_ID:-skipped}"

step corpus
CORPUS_KEY="${PREFIX}${CENTRAL_CORPUS_KEY}"
if skip_allowed && s3_exists "$MODEL_BUCKET" "$CORPUS_KEY"; then
  echo "corpus exists; skipping the export"
else
  python3 "${CODEBUILD_SRC_DIR}/examples/k8s-autoscaling/demo/scripts/index-knowledge.py" \
    --export-only --strict --s3-bucket="${MODEL_BUCKET}" --s3-key="${CORPUS_KEY}"
fi

step llama
ecr_login "$REGISTRY"
retry 5 0 crane copy "$CENTRAL_LLAMA_UPSTREAM" "${REGISTRY}/${CENTRAL_LLAMA_REPO}:${CENTRAL_LLAMA_TAG}"

step wait-amd64
wait_build "$AMD_ID" amd64

step gguf
gguf_track > "$CENTRAL_TMP/gguf-track.log" 2>&1 &
GGUF_PID=$!
echo "GGUF track started in the background (log in gguf-track.log)"

step wait-arm64
wait_build "$ARM_ID" arm64

step join
ecr_login "$REGISTRY"
for repo in $CENTRAL_BUILT_REPOS; do
  retry 5 0 crane index append -t "${REGISTRY}/${repo}:${C12}" \
    -m "${REGISTRY}/${repo}:${C12}-amd64" -m "${REGISTRY}/${repo}:${C12}-arm64" \
    || { echo "index append failed for ${repo}"; exit 1; }
done
DIGESTS="$CENTRAL_TMP/digests"
: > "$DIGESTS"
for repo in $CENTRAL_BUILT_REPOS $CENTRAL_LLAMA_REPO; do
  tag="$C12"
  [ "$repo" = "$CENTRAL_LLAMA_REPO" ] && tag="$CENTRAL_LLAMA_TAG"
  digest=$(retry 5 0 crane digest "${REGISTRY}/${repo}:${tag}")
  echo "${repo} ${digest}" >> "$DIGESTS"
done
llama_digest=$(awk -v r="$CENTRAL_LLAMA_REPO" '$1 == r {print $2}' "$DIGESTS")
if [ "$llama_digest" != "$CENTRAL_LLAMA_DIGEST" ]; then
  echo "llama-cpp digest ${llama_digest} is not the pinned ${CENTRAL_LLAMA_DIGEST}"
  exit 1
fi

step images-ready
retry 5 0 aws s3 cp --quiet "s3://${MODEL_BUCKET}/${CORPUS_KEY}" "$CENTRAL_TMP/corpus.jsonl"
python3 - "$DIGESTS" "$CENTRAL_TMP/corpus.jsonl" "$CORPUS_KEY" "$CENTRAL_TMP/images-ready.json" <<'PY'
import json, os, re, sys
digests, corpus, key, out = sys.argv[1:5]
images = {}
for line in open(digests):
    repo, digest = line.split()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        sys.exit("digest for %s is not sha256:<64 hex>" % repo)
    images[repo] = digest
if len(images) != 8:
    sys.exit("expected 8 images, got %d" % len(images))
size = os.path.getsize(corpus)
with open(corpus, "rb") as f:
    chunks = sum(1 for line in f if line.strip())
if size <= 0 or chunks <= 0:
    sys.exit("the corpus is empty")
marker = {
    "version": 1,
    "tag": os.environ["RELEASE_TAG"],
    "commit": os.environ["CODEBUILD_RESOLVED_SOURCE_VERSION"],
    "images": images,
    "knowledge": {"key": key, "bytes": size, "chunks": chunks},
}
json.dump(marker, open(out, "w"), indent=1, sort_keys=True)
PY
retry 5 0 aws s3 cp --quiet --content-type application/json "$CENTRAL_TMP/images-ready.json" "s3://${MODEL_BUCKET}/${PREFIX}images-ready.json"
echo "wrote ${PREFIX}images-ready.json"

step wait-gguf
gguf_rc=0
wait "$GGUF_PID" || gguf_rc=$?
GGUF_PID=""
cat "$CENTRAL_TMP/gguf-track.log"
if [ "$gguf_rc" -ne 0 ]; then
  echo "the GGUF track failed"
  exit 1
fi

step ready
GGUF_BYTES=$(s3_size "$MODEL_BUCKET" "${PREFIX}${CENTRAL_GGUF_KEY}")
python3 - "$CENTRAL_TMP/ready.json" "${PREFIX}${CENTRAL_GGUF_KEY}" "$GGUF_BYTES" <<'PY'
import datetime, json, os, sys
out, key, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
if size <= 0:
    sys.exit("the GGUF is empty")
marker = {
    "version": 1,
    "tag": os.environ["RELEASE_TAG"],
    "commit": os.environ["CODEBUILD_RESOLVED_SOURCE_VERSION"],
    "gguf": {"key": key, "bytes": size},
    "buildId": os.environ.get("CODEBUILD_BUILD_ID", ""),
    "completedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
json.dump(marker, open(out, "w"), indent=1, sort_keys=True)
PY
retry 5 0 aws s3 cp --quiet --content-type application/json "$CENTRAL_TMP/ready.json" "s3://${MODEL_BUCKET}/${PREFIX}ready.json"
echo "wrote ${PREFIX}ready.json; prep complete"
