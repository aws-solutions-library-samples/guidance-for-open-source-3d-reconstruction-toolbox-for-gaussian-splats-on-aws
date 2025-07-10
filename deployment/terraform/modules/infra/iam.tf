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

# Create data block for Lambda trusted entity policy
data "aws_iam_policy_document" "trusted_entity_lambda" {
  statement {
    effect = "Allow"
    principals {
      type = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

# Create data block for Step Functions trusted entity policy
data "aws_iam_policy_document" "trusted_entity_step_functions" {
  statement {
    effect = "Allow"
    principals {
      type = "Service"
      identifiers = ["states.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

# Create data block for Container Sagemaker/EC2 trusted entity policy
data "aws_iam_policy_document" "trusted_entity_container" {
  statement {
    effect = "Allow"
    principals {
      type = "Service"
      identifiers = [
          "ec2.amazonaws.com",
          "sagemaker.amazonaws.com"
        ]
    }
    actions = ["sts:AssumeRole"]
  }
}

# Create data block for S3 Policy
data "aws_iam_policy_document" "iam_policy_s3" {
  statement {
    effect = "Allow"
    actions = [
        "s3:Abort*",
        "s3:DeleteObject*",
        "s3:GetBucket*",
        "s3:GetObject*",
        "s3:List*",
        "s3:ListBucket",
        "s3:PutObject",
        "s3:PutObjectLegalHold",
        "s3:PutObjectRetention",
        "s3:PutObjectTagging",
        "s3:PutObjectVersionTagging"
        ]
    resources = [
        "${aws_s3_bucket.s3_bucket.arn}",
        "${aws_s3_bucket.s3_bucket.arn}/*"
        ]
  }
}

# IAM policy for "S3" access
resource "aws_iam_policy" "iam_policy_s3" {
  name = "${var.project_prefix}-policy-s3-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_s3.json
depends_on = [
  aws_s3_bucket.s3_bucket
  ]
}

data "aws_iam_policy_document" "iam_policy_eventbridge" {
  statement {
    effect = "Allow"
    actions = [
      "events:DescribeRule",
      "events:PutRule",
      "events:PutTargets",
      "events:DeleteRule",
      "events:RemoveTargets",
      "events:EnableRule",
      "events:DisableRule"
    ]
    resources = [
      "arn:aws:events:*:*:rule/${var.project_prefix}-*",
      "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
      "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule",
      "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTuningJobsRule",
      "arn:aws:events:*:*:rule/StepFunctionsGetEventsForECSTaskRule",
      "arn:aws:events:*:*:rule/StepFunctionsGetEventsForBatchJobsRule"
    ]
  }
  statement {
    effect = "Allow"
    actions = [
      "events:CreateRule",
      "events:DescribeRule",
      "events:PutRule",
      "events:PutTargets"
    ]
    resources = [
      "arn:aws:events:*:*:rule/StepFunctions*"
    ]
  }
}

# IAM policy for "S3" access
resource "aws_iam_policy" "iam_policy_eventbridge" {
  name = "${var.project_prefix}-policy-event-bridge-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_eventbridge.json
}

# Create data block for DynamoDB Policy
data "aws_iam_policy_document" "iam_policy_dynamodb" {
  statement {
    effect = "Allow"
    actions = [
        "dynamodb:BatchGetItem",
        "dynamodb:BatchWriteItem",
        "dynamodb:ConditionCheckItem",
        "dynamodb:DeleteItem",
        "dynamodb:DescribeTable",
        "dynamodb:GetItem",
        "dynamodb:GetRecords",
        "dynamodb:GetShardIterator",
        "dynamodb:PutItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:UpdateItem"
        ]
    resources = [
        "${aws_dynamodb_table.ddb_table.arn}"
        ]
  }
}

# IAM policy for "DynamoDB" access
resource "aws_iam_policy" "iam_policy_dynamodb" {
  name = "${var.project_prefix}-policy-dynamodb-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_dynamodb.json
depends_on = [
  aws_dynamodb_table.ddb_table
  ]
}

# Create data block for ECR Policy
data "aws_iam_policy_document" "iam_policy_ecr" {
  statement {
    effect = "Allow"
    actions = [
          "ecr:GetAuthorizationToken"
        ]
    resources = [
        "*"
        ]
  }
  statement {
    effect = "Allow"
    actions = [
          "ecr:CreateRepository"
        ]
    resources = [
        "arn:aws:ecr:*:*:repository/${var.project_prefix}-*"
        ]
  }
  statement {
    effect = "Allow"
    actions = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:GetDownloadUrlForLayer",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart"
        ]
    resources = [
          "${aws_ecr_repository.ecr_repo.arn}"
        ]
  }
}

# IAM policy for ECR access
resource "aws_iam_policy" "iam_policy_ecr" {
  name = "${var.project_prefix}-policy-ecr-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_ecr.json
}

# Create data block for Step Functions Policy
data "aws_iam_policy_document" "iam_policy_step_functions" {
  statement {
    effect = "Allow"
    actions = [
        "states:StartExecution"
        ]
    resources = [
        "arn:aws:states:*:*:stateMachine:${var.project_prefix}-*"
        ]
  }
}

# IAM policy for "Step Functions" access
resource "aws_iam_policy" "iam_policy_stepfunctions" {
  name = "${var.project_prefix}-policy-stepfunctions-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_step_functions.json
}

# Create data block for SageMaker Policy
data "aws_iam_policy_document" "iam_policy_sagemaker" {
  statement {
    effect = "Allow"
    actions = [
        "sagemaker:DescribeTransformJob",
        "sagemaker:CreateTransformJob",
        "sagemaker:AddTags"
    ]
    resources = [
        "arn:aws:sagemaker:*:*:transform-job/${var.project_prefix}-*"
    ]
  }
}

# IAM policy for "SageMaker" access
resource "aws_iam_policy" "iam_policy_sagemaker" {
  name = "${var.project_prefix}-policy-sagemaker-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_sagemaker.json
}

# Create data block for SSM
data "aws_iam_policy_document" "iam_policy_ssm" {
  statement {
    effect = "Allow"
    actions = [
        "ssm:GetParameters",
        "ssm:GetParameter",
        "ssm:DescribeParameter",
        "ssm:GetParameterHistory"
    ]
    resources = [
        "arn:aws:ssm:*:*:parameter/${var.project_prefix}-*"
    ]
  }
}

# IAM policy for "SSM" access
resource "aws_iam_policy" "iam_policy_ssm" {
  name = "${var.project_prefix}-policy-ssm-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_ssm.json
}

# Create data block for SNS
data "aws_iam_policy_document" "iam_policy_sns_doc" {
  statement {
    effect = "Allow"
    actions = [
        "sns:Publish"
    ]
    resources = [
        "${aws_sns_topic.sns_topic.arn}"
        ]
  }
}

# IAM policy for SNS access
resource "aws_iam_policy" "iam_policy_sns" {
  name = "${var.project_prefix}-policy-sns-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_sns_doc.json
}

# Create data block for Step Function Logging
data "aws_iam_policy_document" "iam_policy_sfn_logs" {
  statement {
    effect = "Allow"
    actions = [
      "logs:DescribeLogGroups"
    ]
    resources = [
      "arn:aws:logs:*:*:log-group:/aws/stepfunctions/${var.project_prefix}-*"
    ]
  }
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = [
      "arn:aws:logs:*:*:log-group:/aws/stepfunctions/${var.project_prefix}-*",
      "arn:aws:logs:*:*:log-group:/aws/stepfunctions/${var.project_prefix}-*:*"
    ]
  }
  statement {
    effect = "Allow"
    actions = [
      "logs:GetLogDelivery",
      "logs:CreateLogDelivery",
      "logs:UpdateLogDelivery",
      "logs:DeleteLogDelivery"
    ]
    resources = [
      "arn:aws:logs:*:*:destination:${var.project_prefix}-*"
    ]
  }
  statement {
    effect = "Allow"
    actions = [
      "logs:DescribeResourcePolicies",
      "logs:PutResourcePolicy"
    ]
    resources = [
      "arn:aws:logs:*:*:destination:${var.project_prefix}-*"
    ]
  }
}

# IAM policy for "Log" access
resource "aws_iam_policy" "iam_policy_step_functions_logs" {
  name = "${var.project_prefix}-policy-step-functions-logs-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_sfn_logs.json
}

resource "aws_iam_policy" "sagemaker_policy" {
  name = "${var.project_prefix}-sagemaker-policy-${var.tf_random_suffix}"
  policy = <<EOT
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeTrainingJob"
            ],
            "Resource": [
                "arn:aws:sagemaker:*:*:training-job/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:StopTrainingJob"
            ],
            "Resource": [
                "arn:aws:sagemaker:*:*:training-job/${var.project_prefix}-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:ListTags"
            ],
            "Resource": [
                "arn:aws:sagemaker:*:*:training-job/${var.project_prefix}-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:AddTags"
            ],
            "Resource": [
                "arn:aws:sagemaker:*:*:training-job/${var.project_prefix}-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": [
                "arn:aws:iam::*:role/${var.project_prefix}-*"
            ],
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "sagemaker.amazonaws.com"
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "events:DescribeRule"
            ],
            "Resource": [
                "arn:aws:events:*:*:rule/${var.project_prefix}-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "events:PutTargets",
                "events:PutRule"
            ],
            "Resource": [
                "arn:aws:events:*:*:rule/${var.project_prefix}-*"
            ]
        }
    ]
}
EOT
}

# Create data block for invoking Lambda
data "aws_iam_policy_document" "iam_policy_invoke_lambda" {
  statement {
    effect = "Allow"
    actions = [
      "lambda:InvokeFunction"
    ]
    resources = [
      "${aws_lambda_function.lambda_workflow_complete.arn}"
      ]
  }
}

# IAM policy for invoking "Lambda"
resource "aws_iam_policy" "iam_policy_invoke_lambda" {
  name = "${var.project_prefix}-policy-invoke-lambda-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_invoke_lambda.json
}

# Create data block for quering cloudwatch logs
data "aws_iam_policy_document" "iam_policy_cloudwatch_lambda" {
  statement {
    effect = "Allow"
    actions = [
      "logs:DescribeLogStreams",
      "logs:GetLogEvents"
    ]
    resources = [
      "arn:aws:logs:*:*:log-group:/aws/lambda/${var.project_prefix}-*",
      "arn:aws:logs:*:*:log-group:/aws/lambda/${var.project_prefix}-*:*",
      "arn:aws:logs:*:*:log-group:/aws/sagemaker/TrainingJobs",
      "arn:aws:logs:*:*:log-group:/aws/sagemaker/TrainingJobs:*"
      ]
  }
}

# IAM policy for quering cloudwatch logs
resource "aws_iam_policy" "iam_policy_cloudwatch_lambda" {
  name = "${var.project_prefix}-policy-cloudwatch-lambda-${var.tf_random_suffix}"
  policy = data.aws_iam_policy_document.iam_policy_cloudwatch_lambda.json
}

# Create data block for AWS managed policy for basic lambda permissions for logging
data "aws_iam_policy" "AWSLambdaBasicExecutionRole_policy" {
  arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Create data block for AWS managed policy for SageMaker Full Access (for container)
data "aws_iam_policy" "AmazonSageMakerFullAccess_policy" {
  arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Local vars needed to apply policies to role in bulk
locals {
    lambda_policies = {
        basic_log = "${data.aws_iam_policy.AWSLambdaBasicExecutionRole_policy.arn}",
        s3 = "${aws_iam_policy.iam_policy_s3.arn}",
        ssm = "${aws_iam_policy.iam_policy_ssm.arn}",
        stepfunctions = "${aws_iam_policy.iam_policy_stepfunctions.arn}",
        dynamodb = "${aws_iam_policy.iam_policy_dynamodb.arn}",
        sns = "${aws_iam_policy.iam_policy_sns.arn}",
        sagemaker = "${aws_iam_policy.sagemaker_policy.arn}",
        cloudwatch = "${aws_iam_policy.iam_policy_cloudwatch_lambda.arn}"
    }
    stepfunctions_policies = {
        basic_log = "${data.aws_iam_policy.AWSLambdaBasicExecutionRole_policy.arn}",
        s3 = "${aws_iam_policy.iam_policy_s3.arn}",
        sagemaker = "${data.aws_iam_policy.AmazonSageMakerFullAccess_policy.arn}",
        event_bridge = "${aws_iam_policy.iam_policy_eventbridge.arn}",
        sfn_logs = "${aws_iam_policy.iam_policy_step_functions_logs.arn}",
        lambda = "${aws_iam_policy.iam_policy_invoke_lambda.arn}"
    }
    container_policies = {
        sagemaker = "${data.aws_iam_policy.AmazonSageMakerFullAccess_policy.arn}",
        ecr = "${aws_iam_policy.iam_policy_ecr.arn}",
        s3 = "${aws_iam_policy.iam_policy_s3.arn}"
    }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
 name = "${var.project_prefix}-lambda-role-${var.tf_random_suffix}"
 assume_role_policy = data.aws_iam_policy_document.trusted_entity_lambda.json
 force_detach_policies = true
}

# Policy Attachments onto the "Lambda" role
resource "aws_iam_role_policy_attachment" "attach_iam_policy_s3_role" {
  for_each = local.lambda_policies
  role = aws_iam_role.lambda_role.name
  policy_arn = each.value
}

# IAM Role for Step Functions
resource "aws_iam_role" "step_functions_role" {
 name = "${var.project_prefix}-step-functions-role-${var.tf_random_suffix}"
 assume_role_policy = data.aws_iam_policy_document.trusted_entity_step_functions.json
 force_detach_policies = true
}

# Policy Attachments onto the "Step Functions" role
resource "aws_iam_role_policy_attachment" "attach_iam_policy_step_function_role" {
  for_each = local.stepfunctions_policies
  role = aws_iam_role.step_functions_role.name
  policy_arn = each.value
}

# IAM Role for Container access
resource "aws_iam_role" "container_role" {
 name = "${var.project_prefix}-container-role-${var.tf_random_suffix}"
 assume_role_policy = data.aws_iam_policy_document.trusted_entity_container.json
 force_detach_policies = true
}

# Policy Attachments onto the Container role
resource "aws_iam_role_policy_attachment" "attach_iam_policy_container_role" {
  for_each = local.container_policies
  role = aws_iam_role.container_role.name
  policy_arn = each.value
}

# SSM Parameter to store the container role arn
resource "aws_ssm_parameter" "parameter_container_role_arn" {
  name = "${var.project_prefix}-container-role-arn-${var.tf_random_suffix}"
  type = "String"
  value = "${aws_iam_role.container_role.arn}"
}
