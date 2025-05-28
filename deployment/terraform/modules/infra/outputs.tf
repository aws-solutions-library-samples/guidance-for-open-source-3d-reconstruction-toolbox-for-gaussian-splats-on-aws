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