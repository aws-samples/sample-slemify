# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# The workshop cluster. Terraform stops at an empty EKS Auto Mode cluster with the
# addons Slemify needs, a model bucket, and the orchestrator's Bedrock role.
# Everything on the cluster (NodePools, OpenSearch, the demo) is applied by
# ../seed/seed.sh, which runs in the same CodeBuild job right after apply.

data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

locals {
  azs               = slice(data.aws_availability_zones.available.names, 0, 3)
  model_bucket_name = var.model_bucket_name != "" ? var.model_bucket_name : "slemify-models-${data.aws_caller_identity.current.account_id}-${var.region}"

  cluster_admin_policy = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"

  # Roles that get cluster-admin access entries. Empty ARNs are dropped so the
  # same code runs inside and outside Workshop Studio.
  access_roles = {
    for k, v in {
      participant = var.participant_role_arn
      ide         = var.ide_role_arn
    } : k => v if v != ""
  }

  tags = {
    Cluster = var.cluster_name
  }
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.7"

  name = var.cluster_name
  cidr = var.vpc_cidr

  azs             = local.azs
  public_subnets  = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 8, k)]
  private_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 3, k + 1)]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = local.tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.25"

  name                   = var.cluster_name
  kubernetes_version     = var.cluster_version
  endpoint_public_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Auto Mode owns node lifecycle, the AMI, and the "default" NodeClass that the
  # Slemify SLM NodePools reference (pkg/serving/nodepool.go, ProvisionerAutoMode).
  # The built-in pools carry the orchestrator, tools, and OpenSearch; the SLM
  # pools (tainted) are applied by the seed step.
  compute_config = {
    enabled    = true
    node_pools = ["system", "general-purpose"]
  }

  # Auto Mode requires the access-entry API; aws-auth is not used.
  authentication_mode                      = "API"
  enable_cluster_creator_admin_permissions = true

  access_entries = {
    for k, arn in local.access_roles : k => {
      principal_arn = arn
      policy_associations = {
        admin = {
          policy_arn   = local.cluster_admin_policy
          access_scope = { type = "cluster" }
        }
      }
    }
  }

  create_cloudwatch_log_group = false

  # Pod Identity is how Slemify jobs and the orchestrator get AWS credentials.
  addons = {
    eks-pod-identity-agent = {}
  }

  tags = local.tags
}
