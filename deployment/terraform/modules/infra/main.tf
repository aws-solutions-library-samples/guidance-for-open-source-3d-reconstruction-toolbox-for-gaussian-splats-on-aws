# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.5.0"
    } 
  }
}

# Create output file for container build script
resource "local_file" "save_output_file" {
  filename = "${path.module}/../../outputs.json"
  content  = jsonencode({
      "${var.project_prefix}-${var.tf_random_suffix}":{
        "BucketName" = "${aws_s3_bucket.s3_bucket.id}"
        "Region" = "${var.region}"
        "ECRRepo" = "${aws_ecr_repository.ecr_repo.repository_url}"
        "ContainerRole" = "${aws_iam_role.container_role.name}"
      }
    }
  )
}