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