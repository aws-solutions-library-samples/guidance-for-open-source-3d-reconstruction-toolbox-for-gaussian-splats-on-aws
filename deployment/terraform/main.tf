# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "~> 5.97.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7.2"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.5.0"
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

/*
# Next, deploy the frontend
# terraform apply -target=module.frontend -var-file=terraform.tfvars
module "frontend" {
  source = "./modules/frontend"
  account_id = var.account_id
  region = var.region
  stage = var.stage
  tf_random_suffix = random_string.tf_random_suffix.result
  depends_on = [ 
    module.infra
    ]
}
*/
