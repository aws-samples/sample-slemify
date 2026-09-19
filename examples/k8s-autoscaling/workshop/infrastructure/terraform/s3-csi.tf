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

resource "aws_eks_addon" "s3_csi" {
  cluster_name = module.eks.cluster_name
  addon_name   = "aws-mountpoint-s3-csi-driver"

  pod_identity_association {
    role_arn        = aws_iam_role.s3_csi.arn
    service_account = "s3-csi-driver-sa"
  }

  tags = local.tags
}
