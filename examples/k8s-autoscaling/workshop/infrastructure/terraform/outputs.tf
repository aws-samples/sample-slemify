# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

output "cluster_name" {
  description = "Name of the EKS Auto Mode cluster."
  value       = module.eks.cluster_name
}

output "region" {
  description = "Region the cluster runs in."
  value       = var.region
}

output "model_bucket" {
  description = "S3 bucket for Slemify state, artifacts, and the analyst GGUF."
  value       = aws_s3_bucket.models.id
}

output "bedrock_region" {
  description = "Region Bedrock calls are routed to."
  value       = var.bedrock_region
}

output "update_kubeconfig_command" {
  description = "Run this to point kubectl at the cluster."
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}"
}
