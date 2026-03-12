# Dedicated G6e compute environment (spot)
resource "aws_batch_compute_environment" "g6e_spot_compute_env" {
  name         = "${var.project_prefix}-g6e-spot-compute-env-${var.tf_random_suffix}-${formatdate("YYYYMMDDhhmmss", timestamp())}"
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
    max_vcpus          = 128
    desired_vcpus      = 0
    instance_type      = ["g6e.4xlarge"]
    
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
    time_sleep.iam_propagation
  ]
}

# Dedicated G6e job queue
resource "aws_batch_job_queue" "g6e_job_queue" {
  name     = "${var.project_prefix}-g6e-job-queue-${var.tf_random_suffix}-${formatdate("YYYYMMDDhhmmss", timestamp())}"
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
