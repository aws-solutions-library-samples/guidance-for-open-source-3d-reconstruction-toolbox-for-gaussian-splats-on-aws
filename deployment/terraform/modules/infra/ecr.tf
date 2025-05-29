# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

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
