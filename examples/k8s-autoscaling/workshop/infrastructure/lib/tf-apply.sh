#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# One terraform apply, run by `retry` in buildspec.yaml (retry needs an
# external command, not a shell function). Arguments go to terraform apply.
# Failure injection: with tf-eks-fail-first (lib/fault.sh), the first
# attempt of a build that targets module.eks exits 1.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=fault.sh
. "${here}/fault.sh"
case " $* " in
  *" -target=module.eks "*)
    marker="${FAULT_TMP}/fault-tf-eks-fail-first"
    if [ ! -e "$marker" ] && fault_on tf-eks-fail-first; then
      : > "$marker"
      echo "FAULT INJECTION: tf-eks-fail-first, this attempt fails on purpose" >&2
      exit 1
    fi
    ;;
esac
exec terraform apply -input=false -auto-approve "$@"
