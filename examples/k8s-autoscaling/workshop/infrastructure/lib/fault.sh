# shellcheck shell=bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Failure injection for test events. Off unless FAULT_INJECTION is set, and
# the published workshop never sets it. Sourced by lib/tf-apply.sh, the seed
# and the buildspecs.
#
# FAULT_INJECTION is a comma-separated list of name[:arg][@share]:
#   tf-eks-fail-first        the first EKS apply attempt of a build exits 1
#   provision-fail-once      the build fails once per account, before the seed.
#                            Nothing tears the account down yet, so Workshop
#                            Studio's retry cannot succeed. Do not use it
#                            until the cleanup on failure exists.
#   no-central-pairs         the team build ignores the central retriever pairs
#   bedrock-unavailable:<s>  the data pipeline gets ServiceUnavailableException
#                            from Bedrock for <s> seconds after the seed starts
# share is a number from 0 to 1 (default 1). A fault fires in an account when
# the first 8 hex digits of sha256(account id), read as a fraction of 16^8, are
# below share, so the same accounts are chosen on every attempt.
#
# A fault marked "once" records that it fired in the SSM parameter
# /slemify/fault/<name>. Nothing creates that parameter in Terraform or
# CloudFormation, so it survives a rollback and the next attempt runs clean.
#
# Every function returns its status; none calls exit. Must run under bash 3.2.
FAULT_TMP="${CENTRAL_TMP:-/tmp}"
# _fault_find <name>
# Returns 0 when FAULT_INJECTION lists the fault. Sets FAULT_ARG and
# FAULT_SHARE. Returns 1 when it is not listed or the share is not a number
# from 0 to 1.
_fault_find() {
  local want=$1 entry share base arg
  [ -n "${FAULT_INJECTION:-}" ] || return 1
  while read -r entry; do
    entry="${entry// /}"
    [ -n "$entry" ] || continue
    share=1
    case "$entry" in
      *@*) share=${entry##*@}; entry=${entry%@*} ;;
    esac
    base=${entry%%:*}
    arg=""
    case "$entry" in
      *:*) arg=${entry#*:} ;;
    esac
    if [ "$base" = "$want" ]; then
      if ! [[ "$share" =~ ^(0|1|0?\.[0-9]+|1\.0+)$ ]]; then
        echo "WARN fault ${want}: share '${share}' is not a number from 0 to 1; the fault is off" >&2
        return 1
      fi
      FAULT_ARG=$arg
      FAULT_SHARE=$share
      return 0
    fi
  done <<EOT
$(printf '%s' "$FAULT_INJECTION" | tr ',' '\n')
EOT
  return 1
}
# _fault_selected <share>: returns 0 when this account is inside the share.
_fault_selected() {
  local share=$1 account
  case "$share" in
    1|1.0*) return 0 ;;
    0) return 1 ;;
  esac
  account=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || return 1
  python3 - "$account" "$share" <<'PY'
import hashlib, sys
h = int(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:8], 16) / float(16 ** 8)
sys.exit(0 if h < float(sys.argv[2]) else 1)
PY
}
# fault_on <name> [once]
# Returns 0 when the fault is listed and this account is selected. With
# "once", also returns 1 when /slemify/fault/<name> exists, and 1 when that
# cannot be read (so a read error never injects a second failure).
fault_on() {
  local name=$1 err="$FAULT_TMP/fault-ssm.err"
  _fault_find "$name" || return 1
  _fault_selected "$FAULT_SHARE" || return 1
  if [ "${2:-}" = once ]; then
    if aws ssm get-parameter --name "/slemify/fault/${name}" >/dev/null 2>"$err"; then
      return 1
    fi
    grep -q ParameterNotFound "$err" || return 1
  fi
  return 0
}
# fault_arg <name>: prints the argument of a listed fault ("600" for
# bedrock-unavailable:600).
fault_arg() {
  _fault_find "$1" || return 1
  printf '%s\n' "$FAULT_ARG"
}
# fault_fired <name>: records that a once fault fired in this account.
fault_fired() {
  aws ssm put-parameter --name "/slemify/fault/$1" --type String --overwrite \
    --value "fired $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null
}
# fault_bedrock_start <model bucket>
# When bedrock-unavailable is on, writes the outage window the data pipeline
# reads at start: <bucket>/k8s-autoscaling/data/_fault/bedrock.json.
FAULT_BEDROCK_WRITTEN=""
fault_bedrock_start() {
  local bucket=$1 secs
  fault_on bedrock-unavailable || return 0
  secs=$(fault_arg bedrock-unavailable)
  if ! [[ "$secs" =~ ^[0-9]{1,5}$ ]]; then
    echo "WARN fault bedrock-unavailable needs a number of seconds (bedrock-unavailable:600); the fault is off" >&2
    return 0
  fi
  printf '{"until": %d}\n' $(($(date +%s) + secs)) > "$FAULT_TMP/fault-bedrock.json"
  aws s3 cp --quiet "$FAULT_TMP/fault-bedrock.json" "s3://${bucket}/k8s-autoscaling/data/_fault/bedrock.json" || return 0
  FAULT_BEDROCK_WRITTEN=1
  echo "FAULT INJECTION: Bedrock calls from the data pipeline fail for the next ${secs}s"
}
# fault_bedrock_stop <model bucket>: removes the window file, if this run wrote one.
fault_bedrock_stop() {
  [ -n "$FAULT_BEDROCK_WRITTEN" ] || return 0
  aws s3 rm --quiet "s3://${1}/k8s-autoscaling/data/_fault/bedrock.json" >/dev/null 2>&1 || true
  FAULT_BEDROCK_WRITTEN=""
}
