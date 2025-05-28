# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# Generate Archive Files for Lambda Code
data "archive_file" "archive_lambda_job_trigger" {
 type = "zip"
 source_dir = "${path.module}/../../../../backend/lambda/workflow_trigger"
 output_path = "${path.module}/../../../../backend/lambda/workflow_trigger/workflow_trigger.zip"
}

data "archive_file" "archive_lambda_job_complete" {
 type = "zip"
 source_dir = "${path.module}/../../../../backend/lambda/workflow_complete"
 output_path = "${path.module}/../../../../backend/lambda/workflow_complete/workflow_complete.zip"
}

# "workflowTrigger" Lambda Function
resource "aws_lambda_function" "lambda_workflow_trigger" {
 function_name = "${var.project_prefix}-workflowTrigger-${var.tf_random_suffix}"
 runtime = "python3.12"
 timeout = 300
 role = aws_iam_role.lambda_role.arn
 handler = "workflow_trigger.lambda_handler"
 filename = "${path.module}/../../../../backend/lambda/workflow_trigger/workflow_trigger.zip"
 reserved_concurrent_executions = 100

 # Enable X-Ray tracing
 tracing_config {
  mode = "Active"
 }

 environment {
  variables = {
    #STATE_MACHINE_ARN = aws_sfn_state_machine.sfn_state_machine.arn,
    STATE_MACHINE_PARAM_NAME =  "${var.project_prefix}-sfn-arn-${var.tf_random_suffix}"
    SNS_TOPIC_ARN = aws_sns_topic.sns_topic.arn,
    LAMBDA_COMPLETE_NAME = aws_lambda_function.lambda_workflow_complete.arn,
    DDB_TABLE_NAME = aws_dynamodb_table.ddb_table.name,
    ECR_IMAGE_URI = aws_ecr_repository.ecr_repo.repository_url,
    CONTAINER_ROLE_NAME = aws_iam_role.container_role.name
    }
  }
}

# "workflowComplete" Lambda Function
resource "aws_lambda_function" "lambda_workflow_complete" {
 function_name = "${var.project_prefix}-workflowComplete-${var.tf_random_suffix}"
 runtime = "python3.12"
 timeout = 300
 role = aws_iam_role.lambda_role.arn
 handler = "workflow_complete.lambda_handler"
 filename = "${path.module}/../../../../backend/lambda/workflow_complete/workflow_complete.zip"
 reserved_concurrent_executions = 100

 # Enable X-Ray tracing
 tracing_config {
  mode = "Active"
 }

 environment {
  variables = {
    SNS_TOPIC_ARN = aws_sns_topic.sns_topic.arn,
    DDB_TABLE_NAME = aws_dynamodb_table.ddb_table.name
    }
  }
}

# Workflow trigger log group
resource "aws_cloudwatch_log_group" "lambda_log_group_workflow_trigger" {
  name = "/aws/lambda/${aws_lambda_function.lambda_workflow_trigger.function_name}"
  retention_in_days = 365
  depends_on = [ 
    aws_s3_bucket.s3_bucket,
    aws_s3_bucket_notification.bucket_notification
   ]
}

# Workflow complete log group
resource "aws_cloudwatch_log_group" "lambda_log_group_workflow_complete" {
  name = "/aws/lambda/${aws_lambda_function.lambda_workflow_complete.function_name}"
  retention_in_days = 365
}

# Permission for S3 Trigger for workflowSubmission
resource "aws_lambda_permission" "allow_bucket" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_workflow_trigger.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.s3_bucket.arn
}