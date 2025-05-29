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
