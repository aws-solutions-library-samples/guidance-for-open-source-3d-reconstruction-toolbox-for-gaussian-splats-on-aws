# NOTE: Be sure to leave the quotes "" around all values you enter below
account_id = ""
region = "us-east-1"
project_prefix = "3dgs"
s3_trigger_key = "workflow-input"
admin_email = ""
maintain_s3_objects_on_stack_deletion = "true"
enable_code_build_container_build = "true"
deployment_phase = "base"  # Options: "base" or "post"
lambda_reserved_concurrency = 10  # Increase for high-throughput deployments; set to -1 for unreserved
batch_max_vcpus = 64              # Increase per compute environment; each g5.4xlarge job uses 16 vCPUs
