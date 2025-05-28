variable "project_prefix" {
  description = "The project prefix to add to resource names."
  default     = "gs-workflow"
  type        = string
}

variable "account_id" {
  description = "The id of the AWS account."
  default     = "012345678901"
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
  description = "The email to use to get notified of complete workflow. New users can be added through SNS Subscription."
  default     = "eecorn@amazon.com"
  type        = string
}

variable "s3_trigger_key" {
  description = "The S3 key to use for submission of the job json file."
  default     = "workflow-input"
  type        = string
}

variable "maintain_s3_objects_on_stack_deletion" {
  description = "Whether to keep S3 objects when the stack is deleted. The bucket will be orphaned from the stack upon stack deletion if set to true."
  default     = "true"
  type        = string
}

variable "deployment_phase" {
  description = "Which deployment phase to run: 'base' for infrastructure or 'post' for container and model deployment"
  default     = "base"
  type        = string
  validation {
    condition     = contains(["base", "post"], var.deployment_phase)
    error_message = "Valid values for deployment_phase are 'base' or 'post'."
  }
}
