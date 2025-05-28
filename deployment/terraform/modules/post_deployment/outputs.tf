output "docker_image_uri" {
  description = "The URI of the pushed Docker image"
  value       = local.ecr_repo_url
}

output "models_s3_uri" {
  description = "The S3 URI where models are uploaded"
  value       = "s3://${local.bucket_name}/models/models.tar.gz"
}
