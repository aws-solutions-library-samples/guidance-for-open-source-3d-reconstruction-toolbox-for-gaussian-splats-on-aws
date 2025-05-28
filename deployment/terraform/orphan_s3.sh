#!/bin/bash

# List all resources in the Terraform state
echo "Listing all S3 bucket resources in Terraform state..."
BUCKET_RESOURCES=$(terraform state list | grep aws_s3_bucket)

# Display found resources
echo "Found the following S3 bucket resources:"
echo "$BUCKET_RESOURCES"

# Remove each S3 bucket resource from state
echo "Removing S3 buckets from Terraform state..."
for resource in $BUCKET_RESOURCES; do
  echo "Removing $resource from state..."
  terraform state rm "$resource"
done

echo "S3 buckets have been removed from Terraform state."
echo "You can now run 'terraform destroy' again."
