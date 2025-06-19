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
      version = "~> 5.99.1"
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
      version = "3.6.1"
    } 
  }
}

provider "aws" {
  region = "${var.region}"
  allowed_account_ids = [
    "${var.account_id}"
  ]
  # Use below fields to input credentials (either hardcode or use credentials file)
  #shared_credentials_files = ["/home/eecorn/.aws/credentials"]
  #access_key = ""
  #secret_key = ""
}

# Random string for unique resource suffix
resource "random_string" "tf_random_suffix" {
  length = 8
  upper = false
  lower  = true
  numeric = true
  special = false
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
  count = var.deployment_phase == "base" ? 1 : 0
}

# Post-deployment for Docker container and model deployment
module "post_deployment" {
  source = "./modules/post_deployment"
  account_id = var.account_id
  region = var.region
  project_prefix = var.project_prefix
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
