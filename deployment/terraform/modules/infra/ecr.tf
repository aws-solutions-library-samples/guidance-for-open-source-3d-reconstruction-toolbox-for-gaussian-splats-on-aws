# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# Gaussian Splat Repo for ECR
resource "aws_ecr_repository" "ecr_repo" {
  name = "${lower(var.project_prefix)}-ecr-repo-${var.tf_random_suffix}"
  image_tag_mutability = "IMMUTABLE"
  force_delete = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

# SSM Parameter to store the ECR Image arn
resource "aws_ssm_parameter" "parameter_ecr_image_arn" {
  name = "${var.project_prefix}-ecr-image-arn-${var.tf_random_suffix}"
  type = "SecureString"
  value = "${aws_ecr_repository.ecr_repo.arn}"
}

# Note: Docker image building and pushing is now handled in the post_deployment module
