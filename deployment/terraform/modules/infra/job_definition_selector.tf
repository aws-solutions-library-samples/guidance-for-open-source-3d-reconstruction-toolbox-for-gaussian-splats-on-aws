# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Job Definition Selector Lambda Function
data "archive_file" "job_definition_selector_zip" {
  type        = "zip"
  source_file = "${path.root}/../../source/lambda/job_definition_selector/job_definition_selector.py"
  output_path = "${path.root}/../../source/lambda/job_definition_selector/job_definition_selector.zip"
}

resource "aws_lambda_function" "job_definition_selector" {
  filename         = data.archive_file.job_definition_selector_zip.output_path
  function_name    = "${var.project_prefix}-job-definition-selector-${var.tf_random_suffix}"
  role            = aws_iam_role.lambda_role.arn
  handler         = "job_definition_selector.lambda_handler"
  source_code_hash = data.archive_file.job_definition_selector_zip.output_base64sha256
  runtime         = "python3.11"
  timeout         = 60

  environment {
    variables = {
      BATCH_JOB_DEFINITION_SMALL     = aws_batch_job_definition.batch_job_definition_small.name
      BATCH_JOB_DEFINITION_MEDIUM    = aws_batch_job_definition.batch_job_definition_medium.name
      BATCH_JOB_DEFINITION_LARGE     = aws_batch_job_definition.batch_job_definition.name
      BATCH_JOB_DEFINITION_XLARGE    = aws_batch_job_definition.batch_job_definition_xlarge.name
      BATCH_JOB_DEFINITION_G5_4XLARGE = aws_batch_job_definition.batch_job_definition_g5_4xlarge.name
      BATCH_JOB_DEFINITION_G5_8XLARGE = aws_batch_job_definition.batch_job_definition_g5_8xlarge.name
      BATCH_JOB_DEFINITION_G6_4XLARGE = aws_batch_job_definition.batch_job_definition_g6_4xlarge.name
      BATCH_JOB_DEFINITION_G6_8XLARGE = aws_batch_job_definition.batch_job_definition_g6_8xlarge.name
      BATCH_JOB_DEFINITION_G6E_4XLARGE = aws_batch_job_definition.batch_job_definition_g6e_4xlarge.name
      BATCH_JOB_DEFINITION_G5_12XLARGE = aws_batch_job_definition.batch_job_definition_g5_12xlarge.name
      BATCH_JOB_QUEUE              = aws_batch_job_queue.batch_job_queue["g5-4xlarge"].arn
      BATCH_JOB_QUEUE_G5_4XLARGE   = aws_batch_job_queue.batch_job_queue["g5-4xlarge"].arn
      BATCH_JOB_QUEUE_G5_8XLARGE   = aws_batch_job_queue.batch_job_queue["g5-8xlarge"].arn
      BATCH_JOB_QUEUE_G5_12XLARGE  = aws_batch_job_queue.batch_job_queue["g5-12xlarge"].arn
      BATCH_JOB_QUEUE_G6_4XLARGE   = aws_batch_job_queue.batch_job_queue["g6-4xlarge"].arn
      BATCH_JOB_QUEUE_G6_8XLARGE   = aws_batch_job_queue.batch_job_queue["g6-8xlarge"].arn
      BATCH_JOB_QUEUE_G6E          = aws_batch_job_queue.g6e_job_queue.arn
    }
  }

  depends_on = [
    aws_iam_role.lambda_role,
    aws_iam_role_policy_attachment.attach_iam_policy_s3_role
  ]
}