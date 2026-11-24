#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# One terraform apply, run by `retry` in buildspec.yaml (retry needs an
# external command, not a shell function). Arguments go to terraform apply.
exec terraform apply -input=false -auto-approve "$@"
