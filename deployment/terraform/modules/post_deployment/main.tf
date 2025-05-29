# Post-deployment module for Docker container build and model deployment

# Post-deployment module for Docker container build and model deployment

terraform {
  required_providers {
    local = {
      source = "hashicorp/local"
      version = "~> 2.4.0"
    }
    aws = {
      source = "hashicorp/aws"
      version = "~> 5.97.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7.0"
    }
  }
}

# Read outputs.json to get resource names from base deployment
data "local_file" "outputs_json" {
  filename = "${path.module}/../../outputs.json"
}

locals {
  outputs = jsondecode(data.local_file.outputs_json.content)
  project_key = keys(local.outputs)[0]
  ecr_repo_url = local.outputs[local.project_key].ECRRepo
  bucket_name = local.outputs[local.project_key].BucketName
  repo_name = split("/", local.ecr_repo_url)[1]
}

# Write updated outputs.json with project prefix for scripts to use
resource "local_file" "updated_outputs" {
  filename = "${path.module}/../../outputs.json"
  content = jsonencode(merge(local.outputs, {
    (local.project_key) = merge(local.outputs[local.project_key], {
      ProjectPrefix = var.project_prefix
    })
  }))
}

# Package the docker image
resource "null_resource" "docker_packaging" {
  triggers = {
    run_at = timestamp()
  }

  provisioner "local-exec" {
    command = "aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${var.account_id}.dkr.ecr.${var.region}.amazonaws.com"
  }

  provisioner "local-exec" {
    command = "docker build -t ${local.repo_name}:latest -f ${path.root}/../../source/container/Dockerfile ${path.root}/../../source/container/"
  }

  provisioner "local-exec" {
    command = "docker tag ${local.repo_name}:latest ${local.ecr_repo_url}:latest"
  }

  provisioner "local-exec" {
    command = "docker push ${local.ecr_repo_url}:latest"
  }
}

# Create IAM role for the Lambda function
resource "aws_iam_role" "model_deployment_lambda_role" {
  name = "${var.project_prefix}-model-deployment-lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Attach policies to the Lambda role
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.model_deployment_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "${var.project_prefix}-lambda-s3-access"
  role = aws_iam_role.model_deployment_lambda_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Effect = "Allow"
        Resource = [
          "arn:aws:s3:::${local.bucket_name}",
          "arn:aws:s3:::${local.bucket_name}/*"
        ]
      }
    ]
  })
}

# Zip the Lambda code
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir = "${path.module}/../../../../source/lambda/model_deployment/"
  output_path = "${path.module}/../../../../source/lambda/model_deployment/model_deployment.zip"
}

# Create Lambda function for model deployment
resource "aws_lambda_function" "model_deployment_lambda" {
  function_name    = "${var.project_prefix}-model-deployment"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  handler          = "model_deployment.handler"
  runtime          = "python3.12"
  role             = aws_iam_role.model_deployment_lambda_role.arn
  timeout          = 900  # 15 minutes
  memory_size      = 3072
  
  environment {
    variables = {
      S3_BUCKET_NAME = local.bucket_name
    }
  }

  ephemeral_storage {
    size = 10240  # 10 GB
  }
}

# Invoke the Lambda function
resource "null_resource" "invoke_lambda" {
  triggers = {
    run_at = timestamp()
  }

  provisioner "local-exec" {
    command = <<-EOT
      echo "Invoking Lambda function to download and upload models..."
      aws lambda invoke \
        --function-name ${aws_lambda_function.model_deployment_lambda.function_name} \
        --region ${var.region} \
        --invocation-type Event \
        --payload '{}' \
        /tmp/lambda_output.json
      
      echo "Lambda invocation started asynchronously. Check CloudWatch logs for details."
      echo "You can monitor the S3 bucket for the models.tar.gz file to appear."
    EOT
  }

  depends_on = [
    aws_lambda_function.model_deployment_lambda,
    null_resource.docker_packaging
  ]
}
