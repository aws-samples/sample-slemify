# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Mountpoint for Amazon S3 CSI driver. The analyst llama.cpp pod mounts the
# ~17 GB GGUF from the model bucket rather than baking it into an image or
# pulling it on every start.

data "aws_iam_policy_document" "s3_csi" {
  statement {
    sid       = "MountpointListBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.models.arn]
  }
  statement {
    sid    = "MountpointObjectRW"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:AbortMultipartUpload",
      "s3:DeleteObject",
    ]
    resources = ["${aws_s3_bucket.models.arn}/*"]
  }
}

resource "aws_iam_role" "s3_csi" {
  name               = "slemify-${var.cluster_name}-s3-csi"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_trust.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "s3_csi" {
  name   = "mountpoint-access"
  role   = aws_iam_role.s3_csi.id
  policy = data.aws_iam_policy_document.s3_csi.json
}

# The addon reports ACTIVE only 8 to 9 minutes after a node exists (its pods
# cannot schedule before then), and nothing in the seed uses it; only module
# 3's analyst does. So the provisioning job creates it AFTER signalling the
# event ready: the foreground apply runs with enable_s3_csi_addon=false (the
# IAM role and policy still apply, they are cheap), and the tail re-applies
# with =true. The addon stays declared and in state either way, so a repair
# `terraform apply` converges it.
variable "enable_s3_csi_addon" {
  description = "Create the Mountpoint S3 CSI addon. Off in the foreground apply, on in the tail, so the event is ready without waiting the ~9 minutes the addon takes to go ACTIVE."
  type        = bool
  default     = true
}

resource "aws_eks_addon" "s3_csi" {
  count        = var.enable_s3_csi_addon ? 1 : 0
  cluster_name = module.eks.cluster_name
  addon_name   = "aws-mountpoint-s3-csi-driver"

  pod_identity_association {
    role_arn        = aws_iam_role.s3_csi.arn
    service_account = "s3-csi-driver-sa"
  }

  tags = local.tags
}
