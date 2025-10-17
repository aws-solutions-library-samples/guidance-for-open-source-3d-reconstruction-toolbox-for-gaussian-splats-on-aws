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

output "s3_bucket_name_workflow" {
  value = "${aws_s3_bucket.s3_bucket.id}"
}

output "region" {
  value = "${var.region}"
}

output "ecr_repo_url" {
  value = "${aws_ecr_repository.ecr_repo.repository_url}"
}

output "container_role_name" {
  value = aws_iam_role.container_role.name
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.ddb_table.name
}

output "batch_job_definition_small" {
  value = aws_batch_job_definition.batch_job_definition_small.name
}

output "batch_job_definition_medium" {
  value = aws_batch_job_definition.batch_job_definition_medium.name
}

output "batch_job_definition_large" {
  value = aws_batch_job_definition.batch_job_definition.name
}

output "batch_job_definition_xlarge" {
  value = aws_batch_job_definition.batch_job_definition_xlarge.name
}

output "batch_job_queue" {
  value = aws_batch_job_queue.batch_job_queue.name
}

output "job_definition_selector_function_name" {
  value = aws_lambda_function.job_definition_selector.function_name
}