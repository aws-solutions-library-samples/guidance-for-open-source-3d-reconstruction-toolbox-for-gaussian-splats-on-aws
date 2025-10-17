# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Get default VPC
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security group for Batch instances
resource "aws_security_group" "batch_security_group" {
  name_prefix = "${var.project_prefix}-batch-sg-"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_prefix}-batch-security-group"
  }
}

# Launch template for Batch instances
resource "aws_launch_template" "batch_launch_template" {
  name_prefix = "${var.project_prefix}-batch-lt-"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 100
      volume_type = "gp3"
      encrypted   = true
      delete_on_termination = true
    }
  }

  user_data = base64encode(<<-EOF
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="

--==MYBOUNDARY==
Content-Type: text/x-shellscript; charset="us-ascii"

#!/bin/bash
mkdir -p /mnt/workspace
chown ecs-agent:ecs-agent /mnt/workspace
chmod 775 /mnt/workspace

--==MYBOUNDARY==--
  EOF
  )

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "3DGS-Batch-Instance"
      Project = "3D-Gaussian-Splatting"
    }
  }
}

# Batch service role
resource "aws_iam_role" "batch_service_role" {
  name = "${var.project_prefix}-batch-service-role-${var.tf_random_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "batch.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_role_policy" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role_policy_attachment" "batch_service_ecs_policy" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

# Additional policy to ensure batch service role has all required permissions
resource "aws_iam_policy" "batch_service_additional_ecs_policy" {
  name        = "${var.project_prefix}-batch-service-additional-ecs-policy-${var.tf_random_suffix}"
  description = "Additional ECS permissions required by AWS Batch service role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "ecs:DescribeClusters",
          "ecs:ListClusters",
          "ecs:CreateCluster",
          "ecs:DeleteCluster",
          "ecs:UpdateCluster",
          "ecs:PutClusterCapacityProviders",
          "ecs:RegisterTaskDefinition",
          "ecs:DeregisterTaskDefinition",
          "ecs:ListTaskDefinitions",
          "ecs:DescribeTaskDefinition",
          "ecs:RunTask",
          "ecs:StopTask",
          "ecs:ListTasks"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_additional_ecs_policy" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = aws_iam_policy.batch_service_additional_ecs_policy.arn
}

# Instance role for Batch instances
resource "aws_iam_role" "batch_instance_role" {
  name = "${var.project_prefix}-batch-instance-role-${var.tf_random_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_instance_role_policy" {
  role       = aws_iam_role.batch_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy" "batch_instance_s3_policy" {
  name = "BatchInstanceS3Policy"
  role = aws_iam_role.batch_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = ["*"]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "batch_instance_profile" {
  name = "${var.project_prefix}-batch-instance-profile-${var.tf_random_suffix}"
  role = aws_iam_role.batch_instance_role.name
}

# Task role for containers
resource "aws_iam_role" "batch_task_role" {
  name = "${var.project_prefix}-batch-task-role-${var.tf_random_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_task_s3_policy" {
  role       = aws_iam_role.batch_task_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "batch_task_logs_policy" {
  role       = aws_iam_role.batch_task_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# Spot fleet role
resource "aws_iam_role" "spot_fleet_role" {
  name = "${var.project_prefix}-spot-fleet-role-${var.tf_random_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "spotfleet.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "spot_fleet_role_policy" {
  role       = aws_iam_role.spot_fleet_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

# Spot compute environment
resource "aws_batch_compute_environment" "spot_compute_env" {
  name         = "${var.project_prefix}-spot-compute-env-${var.tf_random_suffix}-${formatdate("YYYYMMDDhhmmss", timestamp())}"
  type         = "MANAGED"
  state        = "ENABLED"
  service_role = aws_iam_role.batch_service_role.arn

  lifecycle {
    create_before_destroy = true
  }

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT"
    min_vcpus          = 0
    max_vcpus          = 512
    desired_vcpus      = 0
    instance_type      = ["g5.4xlarge", "g5.2xlarge", "g6.4xlarge", "g6.2xlarge"]
    
    bid_percentage     = 50

    ec2_configuration {
      image_type = "ECS_AL2"
    }

    subnets            = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.batch_security_group.id]
    instance_role      = aws_iam_instance_profile.batch_instance_profile.arn

    launch_template {
      launch_template_id = aws_launch_template.batch_launch_template.id
      version           = "$Latest"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.batch_service_role_policy,
    aws_iam_role_policy_attachment.spot_fleet_role_policy
  ]
}

# On-demand compute environment
resource "aws_batch_compute_environment" "on_demand_compute_env" {
  name         = "${var.project_prefix}-on-demand-compute-env-${var.tf_random_suffix}-${formatdate("YYYYMMDDhhmmss", timestamp())}"
  type         = "MANAGED"
  state        = "ENABLED"
  service_role = aws_iam_role.batch_service_role.arn

  lifecycle {
    create_before_destroy = true
  }

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus          = 0
    max_vcpus          = 128
    desired_vcpus      = 0
    instance_type      = ["g5.4xlarge", "g5.2xlarge", "g6.4xlarge", "g6.2xlarge"]

    ec2_configuration {
      image_type = "ECS_AL2"
    }

    subnets            = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.batch_security_group.id]
    instance_role      = aws_iam_instance_profile.batch_instance_profile.arn

    launch_template {
      launch_template_id = aws_launch_template.batch_launch_template.id
      version           = "$Latest"
    }
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service_role_policy]
}

# Job queue
resource "aws_batch_job_queue" "batch_job_queue" {
  name     = "${var.project_prefix}-job-queue-${var.tf_random_suffix}-${formatdate("YYYYMMDDhhmmss", timestamp())}"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot_compute_env.arn
  }

  compute_environment_order {
    order               = 2
    compute_environment = aws_batch_compute_environment.on_demand_compute_env.arn
  }

  depends_on = [
    aws_batch_compute_environment.spot_compute_env,
    aws_batch_compute_environment.on_demand_compute_env
  ]
}

# Job definitions for different instance types
# Small instances (4 vCPUs)
resource "aws_batch_job_definition" "batch_job_definition_small" {
  name = "${var.project_prefix}-job-definition-small-${var.tf_random_suffix}"
  type = "container"

  container_properties = jsonencode({
    image  = aws_ecr_repository.ecr_repo.repository_url
    vcpus  = 4
    memory = 15000
    jobRoleArn = aws_iam_role.batch_task_role.arn
    command = ["python", "/opt/ml/code/main.py"]
    
    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      }
    ]
    
    mountPoints = [
      {
        sourceVolume  = "workspace"
        containerPath = "/tmp"
        readOnly      = false
      },
      {
        sourceVolume  = "shm"
        containerPath = "/dev/shm"
        readOnly      = false
      }
    ]
    
    volumes = [
      {
        name = "workspace"
        host = {
          sourcePath = "/mnt/workspace"
        }
      },
      {
        name = "shm"
        host = {
          sourcePath = "/dev/shm"
        }
      }
    ]
    
    ulimits = [
      {
        name      = "memlock"
        softLimit = -1
        hardLimit = -1
      },
      {
        name      = "stack"
        softLimit = 67108864
        hardLimit = 67108864
      }
    ]
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 259200
  }
}

# Medium instances (8 vCPUs)
resource "aws_batch_job_definition" "batch_job_definition_medium" {
  name = "${var.project_prefix}-job-definition-medium-${var.tf_random_suffix}"
  type = "container"

  container_properties = jsonencode({
    image  = aws_ecr_repository.ecr_repo.repository_url
    vcpus  = 8
    memory = 30000
    jobRoleArn = aws_iam_role.batch_task_role.arn
    command = ["python", "/opt/ml/code/main.py"]
    
    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      }
    ]
    
    mountPoints = [
      {
        sourceVolume  = "workspace"
        containerPath = "/tmp"
        readOnly      = false
      },
      {
        sourceVolume  = "shm"
        containerPath = "/dev/shm"
        readOnly      = false
      }
    ]
    
    volumes = [
      {
        name = "workspace"
        host = {
          sourcePath = "/mnt/workspace"
        }
      },
      {
        name = "shm"
        host = {
          sourcePath = "/dev/shm"
        }
      }
    ]
    
    ulimits = [
      {
        name      = "memlock"
        softLimit = -1
        hardLimit = -1
      },
      {
        name      = "stack"
        softLimit = 67108864
        hardLimit = 67108864
      }
    ]
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 259200
  }
}

# Large instances (16 vCPUs) - Default
resource "aws_batch_job_definition" "batch_job_definition" {
  name = "${var.project_prefix}-job-definition-${var.tf_random_suffix}"
  type = "container"

  container_properties = jsonencode({
    image  = aws_ecr_repository.ecr_repo.repository_url
    vcpus  = 16
    memory = 60000
    jobRoleArn = aws_iam_role.batch_task_role.arn
    command = ["python", "/opt/ml/code/main.py"]
    
    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      }
    ]
    
    mountPoints = [
      {
        sourceVolume  = "workspace"
        containerPath = "/tmp"
        readOnly      = false
      },
      {
        sourceVolume  = "shm"
        containerPath = "/dev/shm"
        readOnly      = false
      }
    ]
    
    volumes = [
      {
        name = "workspace"
        host = {
          sourcePath = "/mnt/workspace"
        }
      },
      {
        name = "shm"
        host = {
          sourcePath = "/dev/shm"
        }
      }
    ]
    
    ulimits = [
      {
        name      = "memlock"
        softLimit = -1
        hardLimit = -1
      },
      {
        name      = "stack"
        softLimit = 67108864
        hardLimit = 67108864
      }
    ]
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 259200
  }
}

# Extra Large instances (32 vCPUs)
resource "aws_batch_job_definition" "batch_job_definition_xlarge" {
  name = "${var.project_prefix}-job-definition-xlarge-${var.tf_random_suffix}"
  type = "container"

  container_properties = jsonencode({
    image  = aws_ecr_repository.ecr_repo.repository_url
    vcpus  = 32
    memory = 120000
    jobRoleArn = aws_iam_role.batch_task_role.arn
    command = ["python", "/opt/ml/code/main.py"]
    
    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      }
    ]
    
    mountPoints = [
      {
        sourceVolume  = "workspace"
        containerPath = "/tmp"
        readOnly      = false
      },
      {
        sourceVolume  = "shm"
        containerPath = "/dev/shm"
        readOnly      = false
      }
    ]
    
    volumes = [
      {
        name = "workspace"
        host = {
          sourcePath = "/mnt/workspace"
        }
      },
      {
        name = "shm"
        host = {
          sourcePath = "/dev/shm"
        }
      }
    ]
    
    ulimits = [
      {
        name      = "memlock"
        softLimit = -1
        hardLimit = -1
      },
      {
        name      = "stack"
        softLimit = 67108864
        hardLimit = 67108864
      }
    ]
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 259200
  }
}

# G5.4xlarge specific job definition as regular container job
resource "aws_batch_job_definition" "batch_job_definition_g5_4xlarge" {
  name = "${var.project_prefix}-job-definition-g5-4xlarge-${var.tf_random_suffix}"
  type = "container"

  container_properties = jsonencode({
    image  = aws_ecr_repository.ecr_repo.repository_url
    vcpus  = 16
    memory = 60000
    jobRoleArn = aws_iam_role.batch_task_role.arn
    command = ["python", "/opt/ml/code/main.py"]
    
    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      }
    ]
    
    mountPoints = [
      {
        sourceVolume  = "workspace"
        containerPath = "/tmp"
        readOnly      = false
      },
      {
        sourceVolume  = "shm"
        containerPath = "/dev/shm"
        readOnly      = false
      }
    ]
    
    volumes = [
      {
        name = "workspace"
        host = {
          sourcePath = "/mnt/workspace"
        }
      },
      {
        name = "shm"
        host = {
          sourcePath = "/dev/shm"
        }
      }
    ]
    
    ulimits = [
      {
        name      = "memlock"
        softLimit = -1
        hardLimit = -1
      },
      {
        name      = "stack"
        softLimit = 67108864
        hardLimit = 67108864
      }
    ]
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 259200
  }
}

# Removed duplicate policy (already defined above)
