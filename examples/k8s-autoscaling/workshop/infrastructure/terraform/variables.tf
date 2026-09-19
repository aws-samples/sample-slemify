# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

variable "region" {
  description = "AWS region to deploy the cluster into."
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS Auto Mode cluster."
  type        = string
  default     = "cmp321"
}

variable "cluster_version" {
  description = "Kubernetes control plane version."
  type        = string
  default     = "1.36"
}

variable "vpc_cidr" {
  description = "CIDR block for the workshop VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "model_bucket_name" {
  description = "Name of the S3 bucket for Slemify state, artifacts, and the analyst GGUF. Empty means a name is generated from the account id and cluster name."
  type        = string
  default     = ""
}

variable "bedrock_region" {
  description = "Region the orchestrator and Slemify call Bedrock in. In Workshop Studio vended accounts Bedrock is enabled only in us-east-1 and us-west-2."
  type        = string
  default     = "us-west-2"
}

# Cluster access. The identity running terraform apply (the CodeBuild role, or
# you locally) is cluster-admin via enable_cluster_creator_admin_permissions.
# These two add the roles an attendee actually uses: the Workshop Studio
# participant role (console, CloudShell) and the IDE instance role (the
# terminal they run kubectl and slemify in). Leave empty to skip.
variable "participant_role_arn" {
  description = "ARN of the Workshop Studio participant role (WSParticipantRole). Empty outside Workshop Studio."
  type        = string
  default     = ""
}

variable "ide_role_arn" {
  description = "ARN of the IDE EC2 instance role created by the CloudFormation stack. Empty when no IDE is deployed."
  type        = string
  default     = ""
}
