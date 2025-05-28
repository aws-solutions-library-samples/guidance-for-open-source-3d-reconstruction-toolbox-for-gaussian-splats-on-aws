# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# Cloudwatch Log Group for Step Functions
resource "aws_cloudwatch_log_group" "step_functions_log_group" {
  name = "/aws/vendedlogs/${var.project_prefix}-state-machine-${var.tf_random_suffix}"
  retention_in_days = 14
}

# Step Functions State Machine
resource "aws_sfn_state_machine" "sfn_state_machine" {
  name = "${var.project_prefix}-state-machine-${var.tf_random_suffix}"
  role_arn = aws_iam_role.step_functions_role.arn
  definition = templatefile("${path.module}/../../../../backend/state-machines/ASLdefinition.json", {})
  logging_configuration {
    include_execution_data = true
    level                  = "ALL"
    log_destination        = "${aws_cloudwatch_log_group.step_functions_log_group.arn}:*"
  }
  tracing_configuration {
    enabled = true
  }
  depends_on = [
    aws_iam_role_policy_attachment.attach_iam_policy_step_function_role,
    aws_lambda_function.lambda_workflow_complete,
  ]
}

# SSM Parameter to store the step function arn
resource "aws_ssm_parameter" "parameter_step_function_arn" {
  name = "${var.project_prefix}-sfn-arn-${var.tf_random_suffix}"
  type = "SecureString"
  value = "${aws_sfn_state_machine.sfn_state_machine.arn}"
}