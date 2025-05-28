# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# Create logging bucket first
resource "aws_s3_bucket" "log_bucket" {
  bucket = "${var.project_prefix}-logs-${var.region}-${var.tf_random_suffix}"
  force_destroy = var.maintain_s3_objects_on_stack_deletion != "true"
  lifecycle {
    prevent_destroy = false
  }
}

# Enable versioning for log bucket
resource "aws_s3_bucket_versioning" "log_bucket_versioning" {
  bucket = aws_s3_bucket.log_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Add lifecycle rules to log bucket
resource "aws_s3_bucket_lifecycle_configuration" "log_bucket_lifecycle" {
  bucket = aws_s3_bucket.log_bucket.id

  rule {
    id     = "log_lifecycle"
    status = "Enabled"
    filter {
      prefix = ""
    }

    transition {
      days          = 90
      storage_class = "INTELLIGENT_TIERING"
    }

    transition {
      days          = 180
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Main bucket
resource "aws_s3_bucket" "s3_bucket" {
  bucket        = "${var.project_prefix}-bucket-${var.region}-${var.tf_random_suffix}"
  force_destroy = var.maintain_s3_objects_on_stack_deletion != "true"
  lifecycle {
    prevent_destroy = false
  }
}

# Enable versioning for main bucket
resource "aws_s3_bucket_versioning" "s3_bucket_versioning" {
  bucket = aws_s3_bucket.s3_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable access logging
resource "aws_s3_bucket_logging" "s3_bucket_logging" {
  bucket = aws_s3_bucket.s3_bucket.id

  target_bucket = aws_s3_bucket.log_bucket.id
  target_prefix = "access-logs/"
}

# Add lifecycle rules to main bucket
resource "aws_s3_bucket_lifecycle_configuration" "s3_bucket_lifecycle" {
  bucket = aws_s3_bucket.s3_bucket.id

  # Rule 1: Transition objects to different storage classes
  rule {
    id     = "TransitionRule"
    status = "Enabled"
    filter {
      prefix = ""
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }

  # Rule 2: Delete old versions of objects
  rule {
    id     = "CleanupOldVersions"
    status = "Enabled"
    filter {
      prefix = ""
    }

    noncurrent_version_transition {
      noncurrent_days = 7
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  # Rule 3: Clean up incomplete multipart uploads
  rule {
    id     = "AbortIncompleteUploads"
    status = "Enabled"
    filter {
      prefix = ""
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Add bucket policy to allow access logging
resource "aws_s3_bucket_policy" "log_bucket_policy" {
  bucket = aws_s3_bucket.log_bucket.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ServerAccessLogsPolicy"
        Effect = "Allow"
        Principal = {
          Service = "logging.s3.amazonaws.com"
        }
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.log_bucket.arn}/access-logs/*"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
          ArnLike = {
            "aws:SourceArn" = aws_s3_bucket.s3_bucket.arn
          }
        }
      }
    ]
  })
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Block public access for both buckets
resource "aws_s3_bucket_public_access_block" "log_bucket_public_access_block" {
  bucket = aws_s3_bucket.log_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "s3_bucket_public_access_block" {
  bucket = aws_s3_bucket.s3_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Create a bucket cors configuration
resource "aws_s3_bucket_cors_configuration" "bucket_cors_config" {
  bucket = aws_s3_bucket.s3_bucket.id
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET","POST","PUT"]
    allowed_origins = ["*"]
    expose_headers = []
    max_age_seconds = 0
  }
}

# Create a Submission key in the bucket for job submissions
resource "aws_s3_object" "workflow_submission_prefix" {
  bucket = aws_s3_bucket.s3_bucket.id
  key = "${var.s3_trigger_key}/"
}

# Create a bucket notification on workflow submission json upload to invoke the job trigger
resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.s3_bucket.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.lambda_workflow_trigger.arn
    events = ["s3:ObjectCreated:Put"]
    filter_prefix = "${var.s3_trigger_key}/"
    filter_suffix = ".json"
  }
  depends_on = [aws_lambda_permission.allow_bucket]
}
