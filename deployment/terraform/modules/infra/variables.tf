# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

variable "project_prefix" {
  description = "The project prefix to add to resource names."
  default     = "gs-workflow"
  type        = string
}

variable "account_id" {
  description = "The id of the AWS account."
  default     = "0123456789012"
  type        = string
}

variable "region" {
  description = "The name of the AWS region."
  default     = "us-east-1"
  type        = string
}

variable "stage" {
  description = "The stage of the environment (e.g. dev, stage, prod)."
  default     = "dev"
  type        = string
}

variable "admin_email" {
  description = "The email to use to initially access the frontend. New users can be added through Cognito user pool"
  default     = "someone@something.com"
  type        = string
}

variable "tf_random_suffix" {
  description = "The random suffix used for unique naming of resources."
  default     = ""
  type        = string
}

variable "s3_trigger_key" {
  description = "The S3 key to use for submission of the job json file."
  default     = "workflow-input"
  type        = string
}

variable "maintain_s3_objects_on_stack_deletion" {
  description = "Whether to maintain S3 objects when the stack is deleted."
  default     = "true"
  type        = string
}
