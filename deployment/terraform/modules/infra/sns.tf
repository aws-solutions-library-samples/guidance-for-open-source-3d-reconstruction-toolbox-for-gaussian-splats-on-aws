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

# Create an SNS topic to subscribe to when workflow is finished
resource "aws_sns_topic" "sns_topic" {
  name = "${var.project_prefix}-sns-topic-${var.tf_random_suffix}"
}

# Subscribe to the sns topic to receive email notification when workflow is complete
resource "aws_sns_topic_subscription" "admin_email_subscription" {
  topic_arn = aws_sns_topic.sns_topic.arn
  protocol  = "email"
  endpoint  = var.admin_email
}

# SSM Parameter to store the sns topic arn
resource "aws_ssm_parameter" "parameter_sns_arn" {
  name = "${var.project_prefix}-sns-topic-arn-${var.tf_random_suffix}"
  type = "SecureString"
  value = "${aws_sns_topic.sns_topic.arn}"
}