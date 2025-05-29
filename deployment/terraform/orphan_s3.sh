#!/bin/bash
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
