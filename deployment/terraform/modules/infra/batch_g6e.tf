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

# Dedicated G6e compute environment (spot)
resource "aws_batch_compute_environment" "g6e_spot_compute_env" {
  name         = "${var.project_prefix}-g6e-spot-compute-env-${var.tf_random_suffix}"
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
    max_vcpus          = var.batch_max_vcpus
    desired_vcpus      = 0
    instance_type      = ["g6e.4xlarge"]
    
    bid_percentage     = 50

    ec2_configuration {
      image_type = "ECS_AL2"
    }

    ec2_configuration {
      image_type = "ECS_AL2023_NVIDIA"
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
    time_sleep.iam_propagation
  ]
}

# Dedicated G6e job queue
resource "aws_batch_job_queue" "g6e_job_queue" {
  name     = "${var.project_prefix}-g6e-job-queue-${var.tf_random_suffix}"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.g6e_spot_compute_env.arn
  }

  depends_on = [
    aws_batch_compute_environment.g6e_spot_compute_env
  ]
}
