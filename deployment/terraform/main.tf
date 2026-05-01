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

terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "~> 6.12.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7.2"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7.1"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.6.2"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.9"
    }
  }
}

provider "aws" {
  region = "${var.region}"
  allowed_account_ids = [
    "${var.account_id}"
  ]
  # Configure AWS credentials via environment variables, IAM roles, or AWS CLI
  # See: https://registry.terraform.io/providers/hashicorp/aws/latest/docs#authentication-and-configuration
}

# Random string for unique resource suffix - not an API key
resource "random_string" "tf_random_suffix" {
  length = 8
  upper = false  # gitleaks:allow
  lower  = true
  numeric = true  # gitleaks:allow - not an API key, just a boolean flag
  special = false # gitleaks:allow
  
  #keepers = {
  # Change this value to force new random suffix
  #  version = "v1"
  #}
}

# Base infrastructure deployment
module "infra" {
  source = "./modules/infra"
  account_id = var.account_id
  region = var.region
  stage = var.stage
  project_prefix = var.project_prefix
  admin_email = var.admin_email
  s3_trigger_key = var.s3_trigger_key
  tf_random_suffix = random_string.tf_random_suffix.result
  maintain_s3_objects_on_stack_deletion = var.maintain_s3_objects_on_stack_deletion
  lambda_reserved_concurrency = var.lambda_reserved_concurrency
  batch_max_vcpus = var.batch_max_vcpus
  count = var.deployment_phase == "base" ? 1 : 0
}

# Generate outputs.json for post-deployment
resource "local_file" "outputs_json" {
  count = var.deployment_phase == "base" ? 1 : 0
  filename = "${path.module}/outputs.json"
  content = jsonencode({
    "${var.project_prefix}-${random_string.tf_random_suffix.result}" = {
      BucketName = module.infra[0].s3_bucket_name_workflow
      ContainerRole = module.infra[0].container_role_name
      ECRRepo = module.infra[0].ecr_repo_url
      Region = var.region
    }
  })
}

# Post-deployment for Docker container and model deployment
module "post_deployment" {
  source = "./modules/post_deployment"
  account_id = var.account_id
  region = var.region
  project_prefix = var.project_prefix
  enable_code_build_container_build = var.enable_code_build_container_build
}

# Adding guidance solution ID via AWS CloudFormation resource
resource "aws_cloudformation_stack" "guidance_deployment_metrics" {
  name = "tracking-stack"
  template_body = jsonencode({
    AWSTemplateFormatVersion = "2010-09-09",
    Description = "Guidance for Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS (SO9142)",
    Resources = {
      EmptyResource = {
        Type = "AWS::CloudFormation::WaitConditionHandle"
      }
    }
  })
}

# Output DynamoDB table name
output "dynamodb_table_name" {
  description = "Name of the DynamoDB table"
  value = var.deployment_phase == "base" ? module.infra[0].dynamodb_table_name : null
}
