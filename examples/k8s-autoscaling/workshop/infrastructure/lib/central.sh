# shellcheck shell=bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Helpers for central mode, sourced by buildspec.yaml (team side) and by
# lib/prep.sh (central prep build). CodeBuild keeps variables between commands
# but not functions, so every command block that calls a helper sources this
# file first.
#
# Central mode: the central account's prep build has already built the images,
# the knowledge corpus and the analyst GGUF once. A team build reads
# /slemify/central (written by the grant function), waits for the prep
# markers, and copies everything into its own account. When the parameter is
# absent the team build takes today's path and builds everything itself.
#
# Every function returns its status; none calls exit or set -e, so callers can
# test the result. Must run under bash 3.2: no associative arrays, no wait -n,
# no mapfile.
#
# State files live in CENTRAL_TMP (default /tmp):
#   build-t0       epoch seconds when the build decided its mode
#   central-mode   "central" or "today"
#   central.env    CENTRAL_ACCOUNT, CENTRAL_REGION, CENTRAL_BUCKET, CENTRAL_REGISTRY
#   images.env     DEMO_IMAGE, RERANKER_IMAGE, KNOWLEDGE_CORPUS for the seed

CENTRAL_TMP="${CENTRAL_TMP:-/tmp}"

# The seven images the image builds produce, plus llama.cpp, which is copied
# from ghcr.io at a pinned digest (src/pkg/serving/deployment.go).
CENTRAL_BUILT_REPOS="slemify/k8s-autoscaling-orchestrator slemify/k8s-autoscaling-reranker slemify/k8s-autoscaling-decision slemify/data-pipeline slemify/classifier-trainer slemify/classifier-serving slemify/gguf-convert"
CENTRAL_LLAMA_REPO="slemify/llama-cpp"
CENTRAL_LLAMA_TAG="server-b12063"
CENTRAL_LLAMA_DIGEST="sha256:10a38e6cc91760ab157f0a0a8546218d35f4a314432a70298abfa3d096829ad7"
# shellcheck disable=SC2034  # used by lib/prep.sh
CENTRAL_LLAMA_UPSTREAM="ghcr.io/ggml-org/llama.cpp@${CENTRAL_LLAMA_DIGEST}"
CENTRAL_GGUF_KEY="models/k8s-autoscaling-analyst/model-q4_k_m.gguf"
CENTRAL_CORPUS_KEY="knowledge/corpus.jsonl"

# retry <attempts> <deadline epoch> <cmd...>
# Runs an external command (not a shell function: each attempt runs under
# timeout) up to <attempts> times. Pauses 30, 60, 120, then 240 s between
# attempts, plus 0 to 10 s of jitter. With a deadline, each attempt gets the
# seconds left, and no attempt starts if its pause would end past the
# deadline. A deadline of 0 means none. Returns 0 on success, otherwise the
# last attempt's status (1 when the deadline passed first).
retry() {
  local attempts=$1 deadline=$2
  shift 2
  local n=1 rc=1 left pause
  local pauses=(30 60 120 240)
  while :; do
    rc=0
    if [ "$deadline" -gt 0 ]; then
      left=$((deadline - $(date +%s)))
      if [ "$left" -le 0 ]; then
        echo "retry: deadline passed before attempt $n of $1" >&2
        return 1
      fi
      timeout "$left" "$@" || rc=$?
    else
      "$@" || rc=$?
    fi
    [ "$rc" -eq 0 ] && return 0
    if [ "$n" -ge "$attempts" ]; then
      echo "retry: $1 $2 failed $n times (exit $rc)" >&2
      return "$rc"
    fi
    pause=${pauses[$((n > 4 ? 3 : n - 1))]}
    pause=$((pause + RANDOM % 11))
    if [ "$deadline" -gt 0 ] && [ $(($(date +%s) + pause)) -ge "$deadline" ]; then
      echo "retry: $1 $2 failed (exit $rc); no time for another attempt before the deadline" >&2
      return "$rc"
    fi
    echo "retry: $1 $2 failed (attempt $n of $attempts, exit $rc), next in ${pause}s" >&2
    sleep "$pause"
    n=$((n + 1))
  done
}

# read_central_param
# Reads and validates /slemify/central (the name is in CENTRAL_PARAM).
# Returns 0 and writes central.env when valid, 2 when CENTRAL_PARAM is empty
# or the parameter does not exist, 1 on any other read error, 3 when the
# value is invalid. Never prints the value.
read_central_param() {
  [ -n "${CENTRAL_PARAM:-}" ] || return 2
  local err="$CENTRAL_TMP/central-param.err" value rc n=1
  while :; do
    rc=0
    value=$(aws ssm get-parameter --name "$CENTRAL_PARAM" --query Parameter.Value --output text 2>"$err") || rc=$?
    [ "$rc" -eq 0 ] && break
    if grep -q ParameterNotFound "$err"; then
      return 2
    fi
    if [ "$n" -ge 3 ]; then
      echo "could not read ${CENTRAL_PARAM}: $(head -c 300 "$err")"
      return 1
    fi
    n=$((n + 1))
    sleep 10
  done
  CENTRAL_VALUE="$value" python3 - "$CENTRAL_TMP/central.env" <<'PY' || return 3
import json, os, re, sys
name = os.environ.get("CENTRAL_PARAM", "/slemify/central")
def bad(why):
    print("%s is invalid: %s" % (name, why))
    sys.exit(3)
try:
    v = json.loads(os.environ["CENTRAL_VALUE"])
except ValueError:
    bad("not JSON")
if not isinstance(v, dict) or v.get("version") != 1:
    bad("version is not 1")
fields = {}
for key in ("account", "region", "bucket", "registry"):
    if not isinstance(v.get(key), str):
        bad("%s is missing" % key)
    fields[key] = v[key]
if not re.fullmatch(r"[0-9]{12}", fields["account"]):
    bad("account is not 12 digits")
if not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-[0-9]", fields["region"]):
    bad("region is not a region name")
if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", fields["bucket"]):
    bad("bucket is not a bucket name")
if fields["registry"] != "%s.dkr.ecr.%s.amazonaws.com/slemify" % (fields["account"], fields["region"]):
    bad("registry is not <account>.dkr.ecr.<region>.amazonaws.com/slemify")
with open(sys.argv[1], "w") as f:
    for key in ("account", "region", "bucket", "registry"):
        f.write("export CENTRAL_%s=%s\n" % (key.upper(), fields[key]))
PY
}

# check_central_bucket
# Returns 1 when the bucket named in the parameter does not exist (a stale
# parameter left by an earlier event), 0 otherwise. A 403 means the grant has
# not reached this account yet; the marker wait covers that.
check_central_bucket() {
  local err="$CENTRAL_TMP/head-bucket.err"
  if aws s3api head-bucket --bucket "$CENTRAL_BUCKET" >/dev/null 2>"$err"; then
    return 0
  fi
  if grep -qE '404|Not Found|NoSuchBucket' "$err"; then
    echo "The central bucket named in ${CENTRAL_PARAM:-/slemify/central} does not exist. The parameter is stale, probably left by an earlier event."
    return 1
  fi
  if grep -qE '403|Forbidden|AccessDenied' "$err"; then
    echo "The central bucket answered 403. The grant may not have reached this account yet; the marker wait continues."
    return 0
  fi
  echo "WARN head-bucket on the central bucket failed: $(head -c 300 "$err")"
  return 0
}

# decide_mode
# Records BUILD_T0 and writes central-mode. Returns 1 when the parameter
# exists but cannot be read or is invalid, or names a missing bucket.
decide_mode() {
  date +%s > "$CENTRAL_TMP/build-t0"
  local rc=0
  read_central_param || rc=$?
  case "$rc" in
    0)
      echo central > "$CENTRAL_TMP/central-mode"
      # shellcheck disable=SC1091
      . "$CENTRAL_TMP/central.env"
      echo "central mode: images, corpus and GGUF are copied from the central account"
      if [ "$CENTRAL_REGION" != "${AWS_REGION:-}" ]; then
        echo "WARN the central account is in ${CENTRAL_REGION} and this build runs in ${AWS_REGION:-<unset>}; copies read across regions"
      fi
      check_central_bucket || return 1
      ;;
    2)
      echo today > "$CENTRAL_TMP/central-mode"
      echo "today's path: no ${CENTRAL_PARAM:-central parameter}; this build makes its own images and GGUF"
      # A recycled account may still hold the mirror from an earlier event.
      aws ssm delete-parameter --name /slemify/mirror >/dev/null 2>&1 || true
      ;;
    *)
      echo "${CENTRAL_PARAM} exists but cannot be used (code ${rc}); failing before the signal"
      return 1
      ;;
  esac
}

# central_load
# Loads central.env and BUILD_T0 and sets PREFIX from RELEASE_TAG and the
# resolved commit. Returns 1 when either is invalid.
central_load() {
  local tag_re='^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$' commit_re='^[0-9a-f]{40}$'
  # shellcheck disable=SC1091
  . "$CENTRAL_TMP/central.env" || return 1
  BUILD_T0=$(cat "$CENTRAL_TMP/build-t0") || return 1
  if ! [[ "${RELEASE_TAG:-}" =~ $tag_re ]]; then
    echo "RELEASE_TAG is not a valid tag"
    return 1
  fi
  if ! [[ "${CODEBUILD_RESOLVED_SOURCE_VERSION:-}" =~ $commit_re ]]; then
    echo "CODEBUILD_RESOLVED_SOURCE_VERSION is not a 40-character commit"
    return 1
  fi
  PREFIX="releases/${RELEASE_TAG}/${CODEBUILD_RESOLVED_SOURCE_VERSION}/"
}

# s3_exists <bucket> <key>
# Returns 0 when the object exists, 1 when it does not. Other errors are
# retried 5 times, 30 s apart, then return 2.
s3_exists() {
  local err="$CENTRAL_TMP/head-object.$$.err" n=1
  while :; do
    if aws s3api head-object --bucket "$1" --key "$2" >/dev/null 2>"$err"; then
      return 0
    fi
    if grep -qE '404|Not Found|NoSuchKey' "$err"; then
      return 1
    fi
    if [ "$n" -ge 5 ]; then
      echo "head-object s3://$1/$2 failed: $(head -c 300 "$err")" >&2
      return 2
    fi
    n=$((n + 1))
    sleep 30
  done
}

# s3_size <bucket> <key>: prints the object's size in bytes.
s3_size() {
  retry 5 "${3:-0}" aws s3api head-object --bucket "$1" --key "$2" --query ContentLength --output text
}

# wait_for_object <s3 uri> <deadline epoch> [validator]
# Polls every 30 s until the object exists (and, when a validator function is
# given, passes "<validator> <file>"). The object is saved to
# CENTRAL_TMP/<its file name>. Logs the key every 5 minutes, with the step and
# log link of a failed.json beside it if there is one. Returns 1 at the
# deadline.
wait_for_object() {
  local uri=$1 deadline=$2 validator=${3:-}
  local dir="${uri%/*}/" out="$CENTRAL_TMP/${uri##*/}" next_log=0 now
  while :; do
    now=$(date +%s)
    if aws s3 cp --quiet "$uri" "$out" >/dev/null 2>&1; then
      if [ -z "$validator" ] || "$validator" "$out"; then
        return 0
      fi
    fi
    if [ "$now" -ge "$next_log" ]; then
      echo "waiting for ${uri#s3://*/} ($(( (deadline - now) / 60 )) min left)"
      if aws s3 cp --quiet "${dir}failed.json" "$CENTRAL_TMP/failed.json" >/dev/null 2>&1; then
        python3 - "$CENTRAL_TMP/failed.json" <<'PY' || true
import json, sys
f = json.load(open(sys.argv[1]))
print("  the central prep build failed at step %s; log: %s" % (f.get("step"), f.get("logUrl")))
PY
      fi
      next_log=$((now + 300))
    fi
    if [ $((now + 30)) -ge "$deadline" ]; then
      echo "gave up waiting for ${uri#s3://*/}: deadline reached"
      return 1
    fi
    sleep 30
  done
}

# check_marker <file> <images|ready>
# Returns 0 when the marker is for this tag and commit and complete, 1 when
# it is not (which callers treat as "not there yet").
check_marker() {
  python3 - "$1" "$2" <<'PY'
import json, os, re, sys
path, kind = sys.argv[1], sys.argv[2]
def no(why):
    print("marker %s: %s" % (os.path.basename(path), why), file=sys.stderr)
    sys.exit(1)
try:
    m = json.load(open(path))
except (OSError, ValueError):
    no("not JSON")
if not isinstance(m, dict) or m.get("version") != 1:
    no("version is not 1")
if m.get("commit") != os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION"):
    no("written for another commit")
if m.get("tag") != os.environ.get("RELEASE_TAG"):
    no("written for another tag")
def positive(section):
    b = (m.get(section) or {}).get("bytes")
    if not isinstance(b, int) or isinstance(b, bool) or b <= 0:
        no("%s.bytes is not above 0" % section)
if kind == "images":
    repos = os.environ["CENTRAL_BUILT_REPOS"].split() + [os.environ["CENTRAL_LLAMA_REPO"]]
    images = m.get("images")
    if not isinstance(images, dict) or sorted(images) != sorted(repos):
        no("images does not list the 8 repositories")
    for repo, digest in images.items():
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            no("digest for %s is not sha256:<64 hex>" % repo)
    positive("knowledge")
elif kind == "ready":
    positive("gguf")
else:
    no("unknown kind %s" % kind)
PY
}
export CENTRAL_BUILT_REPOS CENTRAL_LLAMA_REPO
marker_images() { check_marker "$1" images; }
marker_ready() { check_marker "$1" ready; }

# marker_field <file> <dotted path>: prints one value from a marker.
marker_field() {
  python3 - "$1" "$2" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    v = v[part]
print(v)
PY
}

# ecr_login <registry host>: logs crane in with this account's ECR token.
ecr_login() {
  local token
  token=$(aws ecr get-login-password --region "${AWS_REGION}") || return 1
  printf '%s' "$token" | crane auth login "$1" --username AWS --password-stdin
}

# login_central <registry host> <region>
# Logs crane in to the central registry with this account's ECR token for
# that region. The central repository policy grants this account.
login_central() {
  local token
  token=$(aws ecr get-login-password --region "$2") || return 1
  printf '%s' "$token" | crane auth login "$1" --username AWS --password-stdin
}

# copy_image <src> <dst> <deadline epoch>
copy_image() {
  retry 5 "$3" crane copy "$1" "$2"
}

# ensure_repo <name>: creates an ECR repository unless it exists.
ensure_repo() {
  aws ecr describe-repositories --repository-names "$1" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$1" \
      --image-scanning-configuration scanOnPush=true >/dev/null 2>&1 || true
}

# copy_object_checked <src uri> <bucket> <key> <bytes> <deadline epoch>
# Server-side copy under retry, then checks the copy's size.
copy_object_checked() {
  local size
  retry 5 "$5" aws s3 cp --quiet "$1" "s3://$2/$3" || return 1
  size=$(s3_size "$2" "$3" "$5") || return 1
  if [ "$size" != "$4" ]; then
    echo "s3://$2/$3 has ${size} bytes, the marker says $4"
    return 1
  fi
}

# central_copy_phase
# Team side, before the full Terraform apply. Waits for images-ready.json,
# copies the eight images and the corpus into this account in parallel,
# writes images.env, and then writes /slemify/mirror so the IDE pulls
# llama.cpp from this account. Returns 1 when any step fails.
central_copy_phase() {
  central_load || return 1
  local wait_deadline=$((BUILD_T0 + ${CENTRAL_WAIT_MINUTES:-90} * 60))
  local copy_deadline=$((BUILD_T0 + ${CENTRAL_COPY_MINUTES:-100} * 60))
  local marker="$CENTRAL_TMP/images-ready.json"
  echo "Waiting for ${PREFIX}images-ready.json from the central prep build"
  wait_for_object "s3://${CENTRAL_BUCKET}/${PREFIX}images-ready.json" "$wait_deadline" marker_images || return 1

  local model_bucket account team_host central_host
  model_bucket=$(terraform output -raw model_bucket) || return 1
  account=$(aws sts get-caller-identity --query Account --output text) || return 1
  team_host="${account}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  central_host="${CENTRAL_REGISTRY%%/*}"

  local repo
  for repo in $CENTRAL_BUILT_REPOS $CENTRAL_LLAMA_REPO; do
    ensure_repo "$repo"
  done
  ecr_login "$team_host" || { echo "crane login to ${team_host} failed"; return 1; }
  login_central "$central_host" "$CENTRAL_REGION" || { echo "crane login to the central registry failed"; return 1; }

  local pids=() names=() digest tag pid i FAILED=0
  for repo in $CENTRAL_BUILT_REPOS $CENTRAL_LLAMA_REPO; do
    digest=$(marker_field "$marker" "images.${repo}") || return 1
    tag=latest
    [ "$repo" = "$CENTRAL_LLAMA_REPO" ] && tag="$CENTRAL_LLAMA_TAG"
    copy_image "${CENTRAL_REGISTRY}/${repo#slemify/}@${digest}" "${team_host}/${repo}:${tag}" "$copy_deadline" \
      > "$CENTRAL_TMP/copy-${repo#slemify/}.log" 2>&1 &
    pids+=("$!")
    names+=("$repo")
  done
  local bytes
  bytes=$(marker_field "$marker" knowledge.bytes) || return 1
  copy_object_checked "s3://${CENTRAL_BUCKET}/${PREFIX}${CENTRAL_CORPUS_KEY}" "$model_bucket" "$CENTRAL_CORPUS_KEY" "$bytes" "$copy_deadline" \
    > "$CENTRAL_TMP/copy-corpus.log" 2>&1 &
  pids+=("$!")
  names+=("corpus")

  i=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      echo "  copied ${names[$i]}"
    else
      FAILED=1
      echo "  copy failed: ${names[$i]}"
      tail -n 20 "$CENTRAL_TMP/copy-${names[$i]#slemify/}.log"
    fi
    i=$((i + 1))
  done
  if [ "$FAILED" -ne 0 ]; then
    echo "central copy failed; see the lines above"
    return 1
  fi

  echo "export DEMO_IMAGE=${team_host}/slemify/k8s-autoscaling-orchestrator:latest RERANKER_IMAGE=${team_host}/slemify/k8s-autoscaling-reranker:latest KNOWLEDGE_CORPUS=s3://${model_bucket}/${CENTRAL_CORPUS_KEY}" \
    > "$CENTRAL_TMP/images.env"
  # Only now does this account hold llama-cpp, so only now may the IDE use it.
  retry 3 "$copy_deadline" aws ssm put-parameter --name /slemify/mirror --type String --overwrite \
    --value "${team_host}/slemify" >/dev/null || { echo "could not write /slemify/mirror"; return 1; }
  echo "images and corpus copied; /slemify/mirror set"
}

# central_gguf_copy <model bucket>
# Waits for ready.json, then copies the GGUF to today's key in the model
# bucket and checks its size.
central_gguf_copy() {
  local deadline=$((BUILD_T0 + ${CENTRAL_TAIL_MINUTES:-140} * 60)) bytes
  wait_for_object "s3://${CENTRAL_BUCKET}/${PREFIX}ready.json" "$deadline" marker_ready || return 1
  bytes=$(marker_field "$CENTRAL_TMP/ready.json" gguf.bytes) || return 1
  copy_object_checked "s3://${CENTRAL_BUCKET}/${PREFIX}${CENTRAL_GGUF_KEY}" "$1" "$CENTRAL_GGUF_KEY" "$bytes" "$deadline"
}

# central_tail
# Team side, after the signal. Copies the GGUF in the background while the
# S3 CSI addon is applied in the foreground, then waits for both. Run from the
# terraform directory. Returns 1 when either fails.
central_tail() {
  central_load || return 1
  local model_bucket gguf_pid addon_rc=0 gguf_rc=0
  model_bucket=$(terraform output -raw model_bucket) || return 1
  echo "Copying the analyst GGUF in the background"
  central_gguf_copy "$model_bucket" > "$CENTRAL_TMP/copy-gguf.log" 2>&1 &
  gguf_pid=$!

  echo "Applying the Mountpoint S3 CSI addon (module 3)"
  local max_retry=3 counter=1
  until terraform apply -input=false -auto-approve -var enable_s3_csi_addon=true
  do
    if [ "$counter" -eq "$max_retry" ]; then
      echo "S3 CSI addon apply failed after ${max_retry} attempts; repair with: make check-infra"
      addon_rc=1
      break
    fi
    echo "addon apply failed, retrying in 30s (attempt $counter of $max_retry)"
    sleep 30
    counter=$((counter + 1))
  done

  wait "$gguf_pid" || gguf_rc=$?
  cat "$CENTRAL_TMP/copy-gguf.log"
  if [ "$gguf_rc" -ne 0 ]; then
    echo "GGUF copy failed; repair with: make check-infra"
  fi
  [ "$addon_rc" -eq 0 ] && [ "$gguf_rc" -eq 0 ]
}
