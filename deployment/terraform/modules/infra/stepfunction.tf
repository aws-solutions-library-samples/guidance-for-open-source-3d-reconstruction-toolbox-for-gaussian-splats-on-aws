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

# Cloudwatch Log Group for Step Functions
resource "aws_cloudwatch_log_group" "step_functions_log_group" {
  name = "/aws/vendedlogs/${var.project_prefix}-state-machine-${var.tf_random_suffix}"
  retention_in_days = 14
}

# Step Functions State Machine
resource "aws_sfn_state_machine" "sfn_state_machine" {
  name = "${var.project_prefix}-state-machine-${var.tf_random_suffix}"
  role_arn = aws_iam_role.step_functions_role.arn
  definition = templatefile("${path.module}/../../../../source/state-machines/ASLdefinition.json", {})
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