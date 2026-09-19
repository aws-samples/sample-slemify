# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.59"
    }
  }

  # The S3 backend is written to backend.tf by buildspec.yaml at run time
  # (bucket and key come from the CloudFormation stack). Local runs without
  # backend.tf use local state.
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project = "slemify-workshop"
    }
  }
}
