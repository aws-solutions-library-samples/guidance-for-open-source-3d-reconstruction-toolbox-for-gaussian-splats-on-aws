variable "account_id" {
  description = "The id of the AWS account."
  type        = string
}

variable "region" {
  description = "The name of the AWS region."
  type        = string
}

variable "project_prefix" {
  description = "The project prefix to add to resource names."
  type        = string
}

# These variables are no longer needed as we're reading from outputs.json
# Keeping the file for reference
