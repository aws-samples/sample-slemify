# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# The orchestrator's Bedrock access. The demo's setup-demo.sh created this role
# and the pod identity association by hand; Terraform owns it so the seed step
# needs no IAM writes. Slemify still creates its own per-project data roles at
# deploy time (see pkg/k8s/identity.go); those are not managed here.

data "aws_iam_policy_document" "pod_identity_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole", "sts:TagSession"]
    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "orchestrator_bedrock" {
  name               = "slemify-${var.cluster_name}-orchestrator-bedrock"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_trust.json
  tags               = local.tags
}

data "aws_iam_policy_document" "orchestrator_bedrock" {
  statement {
    sid    = "InvokeBedrock"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    # Bedrock model ARNs and the us. inference-profile ARNs are not known until
    # the model IDs are pinned; the monolith baseline calls one model plus Titan
    # embeddings. Scope to the account's foundation models and inference profiles.
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*",
    ]
  }
}

resource "aws_iam_role_policy" "orchestrator_bedrock" {
  name   = "bedrock-access"
  role   = aws_iam_role.orchestrator_bedrock.id
  policy = data.aws_iam_policy_document.orchestrator_bedrock.json
}

# Associate the role with the orchestrator service account the demo manifest
# declares (k8s-autoscaling-orchestrator in the slemify namespace). The
# namespace and service account are created by the seed step before the pods
# start; the association can exist ahead of them.
resource "aws_eks_pod_identity_association" "orchestrator_bedrock" {
  cluster_name    = module.eks.cluster_name
  namespace       = "slemify"
  service_account = "k8s-autoscaling-orchestrator"
  role_arn        = aws_iam_role.orchestrator_bedrock.arn
  tags            = local.tags
}
