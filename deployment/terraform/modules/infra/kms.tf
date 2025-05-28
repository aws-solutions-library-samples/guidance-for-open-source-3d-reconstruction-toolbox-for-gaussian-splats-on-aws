# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# Create data block for KMS Policy
data "aws_iam_policy_document" "iam_policy_kms" {
  statement {
    effect = "Allow"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey*",
      "kms:ReEncrypt*"
    ]
    resources = ["*"]  # You can restrict this to specific KMS keys if known
  }
}

# IAM policy for KMS access
resource "aws_iam_policy" "iam_policy_kms" {
  name   = "${var.project_prefix}-policy-kms-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_kms.json
}
