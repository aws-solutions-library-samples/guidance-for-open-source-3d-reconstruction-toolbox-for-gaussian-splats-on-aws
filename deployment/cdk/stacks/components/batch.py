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

"""AWS Batch construct for spot instance support"""

from aws_cdk import (
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_ecs as ecs,
    Environment,
    CfnOutput
)
from constructs import Construct

class BatchConstruct(Construct):
    """Class for AWS Batch Construct"""
    
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            ecr_repo_uri: str,
            container_role_arn: str,
            random_id: str,
            vpc=None,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        self.env = env
        self.ecr_repo_uri = ecr_repo_uri
        self.container_role_arn = container_role_arn
        self.random_id = random_id
        
        # Use provided VPC or create default VPC
        self.vpc = vpc or ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)
        
        # Create Batch service role
        self.batch_service_role = self._create_batch_service_role()
        
        # Create spot fleet role
        self.spot_fleet_role = self._create_spot_fleet_role()
        
        # Create instance role and profile
        self.instance_role, self.instance_profile = self._create_instance_role()
        
        # Create security group once
        self.security_group = self._create_security_group()
        
        # Create launch template with increased volume size
        self.launch_template = self._create_launch_template()
        
        # Create compute environments
        self.spot_compute_env = self._create_spot_compute_environment()
        self.on_demand_compute_env = self._create_on_demand_compute_environment()
        self.g6e_spot_compute_env = self._create_g6e_spot_compute_environment()
        
        # Create job queues
        self.job_queue = self._create_job_queue()
        self.g6e_job_queue = self._create_g6e_job_queue()
        
        # Create job definitions for different instance types
        self.job_definitions = self._create_job_definitions()
        
        # Outputs
        CfnOutput(self, "BatchJobQueueArn", value=self.job_queue.ref)
        CfnOutput(self, "G6eBatchJobQueueArn", value=self.g6e_job_queue.ref)
        for instance_type, job_def in self.job_definitions.items():
            CfnOutput(self, f"BatchJobDefinition{instance_type.replace('.', '').upper()}Arn", value=job_def.ref)

    def _create_batch_service_role(self):
        """Create IAM role for Batch service"""
        role = iam.Role(
            self, "BatchServiceRole",
            assumed_by=iam.ServicePrincipal("batch.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSBatchServiceRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonECS_FullAccess")
            ]
        )
        
        # Add additional ECS permissions like Terraform
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
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
                ],
                resources=["*"]
            )
        )
        
        return role
    
    def _create_spot_fleet_role(self):
        """Create IAM role for EC2 Spot Fleet"""
        return iam.Role(
            self, "SpotFleetRole",
            role_name=f"aws-ec2-spot-fleet-tagging-role-{self.random_id}",
            assumed_by=iam.ServicePrincipal("spotfleet.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2SpotFleetTaggingRole")
            ]
        )

    def _create_instance_role(self):
        """Create IAM role and instance profile for Batch instances"""
        instance_role = iam.Role(
            self, "BatchInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role")
            ]
        )
        
        # Add additional permissions for S3 access
        instance_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                resources=["*"]
            )
        )
        
        # Create task role for containers
        self.task_role = iam.Role(
            self, "BatchTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess")
            ]
        )
        
        instance_profile = iam.CfnInstanceProfile(
            self, "BatchInstanceProfile",
            roles=[instance_role.role_name]
        )
        
        return instance_role, instance_profile

    def _create_launch_template(self):
        """Create launch template with increased volume size"""
        # Create properly formatted MIME user data that matches Terraform
        import base64
        user_data = """MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="

--==MYBOUNDARY==
Content-Type: text/x-shellscript; charset="us-ascii"

#!/bin/bash
mkdir -p /mnt/workspace
chown ecs-agent:ecs-agent /mnt/workspace
chmod 775 /mnt/workspace

--==MYBOUNDARY==--
"""
        encoded_user_data = base64.b64encode(user_data.encode()).decode()
        
        return ec2.CfnLaunchTemplate(
            self, "BatchLaunchTemplate",
            launch_template_data={
                "blockDeviceMappings": [
                    {
                        "deviceName": "/dev/xvda",
                        "ebs": {
                            "volumeSize": 100,
                            "volumeType": "gp3",
                            "encrypted": True,
                            "deleteOnTermination": True
                        }
                    }
                ],
                "userData": encoded_user_data,
                "tagSpecifications": [
                    {
                        "resourceType": "instance",
                        "tags": [
                            {
                                "key": "Name",
                                "value": "3DGS-Batch-Instance"
                            },
                            {
                                "key": "Project",
                                "value": "3D-Gaussian-Splatting"
                            }
                        ]
                    }
                ]
            }
        )

    def _create_spot_compute_environment(self):
        """Create spot compute environment"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        spot_env = batch.CfnComputeEnvironment(
            self, "SpotComputeEnvironment",
            compute_environment_name=f"SpotComputeEnv-{self.random_id}-{timestamp}",
            type="MANAGED",
            state="ENABLED",
            service_role=self.batch_service_role.role_arn,
            compute_resources={
                "type": "EC2",
                "allocationStrategy": "BEST_FIT",
                "bidPercentage": 50,
                "minvCpus": 0,
                "maxvCpus": 512,
                "desiredvCpus": 0,
                "instanceTypes": ["g5.4xlarge", "g5.8xlarge", "g6.4xlarge", "g6.8xlarge", "g6e.4xlarge"],
                "ec2Configuration": [{
                    "imageType": "ECS_AL2"
                }],
                "subnets": [subnet.subnet_id for subnet in self.vpc.public_subnets],
                "securityGroupIds": [self.security_group.security_group_id],
                "instanceRole": self.instance_profile.attr_arn,
                "launchTemplate": {
                    "launchTemplateId": self.launch_template.ref,
                    "version": "$Latest"
                }
            }
        )
        return spot_env
    
    def _create_g6e_spot_compute_environment(self):
        """Create dedicated g6e spot compute environment"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        g6e_spot_env = batch.CfnComputeEnvironment(
            self, "G6eSpotComputeEnvironment",
            compute_environment_name=f"G6eSpotComputeEnv-{self.random_id}-{timestamp}",
            type="MANAGED",
            state="ENABLED",
            service_role=self.batch_service_role.role_arn,
            compute_resources={
                "type": "EC2",
                "allocationStrategy": "BEST_FIT",
                "bidPercentage": 50,
                "minvCpus": 0,
                "maxvCpus": 128,
                "desiredvCpus": 0,
                "instanceTypes": ["g6e.4xlarge"],
                "ec2Configuration": [{
                    "imageType": "ECS_AL2"
                }],
                "subnets": [subnet.subnet_id for subnet in self.vpc.public_subnets],
                "securityGroupIds": [self.security_group.security_group_id],
                "instanceRole": self.instance_profile.attr_arn,
                "launchTemplate": {
                    "launchTemplateId": self.launch_template.ref,
                    "version": "$Latest"
                }
            }
        )
        return g6e_spot_env

    def _create_on_demand_compute_environment(self):
        """Create on-demand compute environment as fallback"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        on_demand_env = batch.CfnComputeEnvironment(
            self, "OnDemandComputeEnvironment",
            compute_environment_name=f"OnDemandComputeEnv-{self.random_id}-{timestamp}",
            type="MANAGED",
            state="ENABLED",
            service_role=self.batch_service_role.role_arn,
            compute_resources={
                "type": "EC2",
                "allocationStrategy": "BEST_FIT_PROGRESSIVE",
                "minvCpus": 0,
                "maxvCpus": 128,
                "desiredvCpus": 0,
                "instanceTypes": ["g5.4xlarge", "g5.8xlarge", "g6.4xlarge", "g6.8xlarge", "g6e.4xlarge"],
                "ec2Configuration": [{
                    "imageType": "ECS_AL2"
                }],
                "subnets": [subnet.subnet_id for subnet in self.vpc.public_subnets],
                "securityGroupIds": [self.security_group.security_group_id],
                "instanceRole": self.instance_profile.attr_arn,
                "launchTemplate": {
                    "launchTemplateId": self.launch_template.ref,
                    "version": "$Latest"
                }
            }
        )
        return on_demand_env

    def _create_security_group(self):
        """Create security group for Batch instances"""
        return ec2.SecurityGroup(
            self, "BatchSecurityGroup",
            vpc=self.vpc,
            description="Security group for Batch compute environment",
            allow_all_outbound=True
        )

    def _create_job_queue(self):
        """Create job queue with spot priority"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        job_queue = batch.CfnJobQueue(
            self, "BatchJobQueue",
            job_queue_name=f"JobQueue-{self.random_id}-{timestamp}",
            state="ENABLED",
            priority=1,
            compute_environment_order=[
                {
                    "order": 1,
                    "computeEnvironment": self.spot_compute_env.ref
                },
                {
                    "order": 2,
                    "computeEnvironment": self.on_demand_compute_env.ref
                }
            ]
        )
        job_queue.add_dependency(self.spot_compute_env)
        job_queue.add_dependency(self.on_demand_compute_env)
        return job_queue
    
    def _create_g6e_job_queue(self):
        """Create dedicated g6e job queue"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        g6e_job_queue = batch.CfnJobQueue(
            self, "G6eBatchJobQueue",
            job_queue_name=f"G6eJobQueue-{self.random_id}-{timestamp}",
            state="ENABLED",
            priority=1,
            compute_environment_order=[
                {
                    "order": 1,
                    "computeEnvironment": self.g6e_spot_compute_env.ref
                }
            ]
        )
        g6e_job_queue.add_dependency(self.g6e_spot_compute_env)
        return g6e_job_queue

    def _create_job_definitions(self):
        """Create job definitions exactly matching Terraform"""
        job_definitions = {}
        
        # Small instances (4 vCPUs) - Terraform: batch_job_definition_small
        job_definitions["small"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionSmall",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 4,
                "memory": 15000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # Medium instances (8 vCPUs) - Terraform: batch_job_definition_medium
        job_definitions["medium"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionMedium",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 8,
                "memory": 30000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # Large instances (16 vCPUs) - Terraform: batch_job_definition (default)
        job_definitions["default"] = batch.CfnJobDefinition(
            self, "BatchJobDefinition",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 16,
                "memory": 60000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # Extra Large instances (32 vCPUs) - Terraform: batch_job_definition_xlarge
        job_definitions["xlarge"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionXlarge",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 32,
                "memory": 400000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # G5.4xlarge specific - Terraform: batch_job_definition_g5_4xlarge
        job_definitions["g5.4xlarge"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionG54xlarge",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 16,
                "memory": 60000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # G5.8xlarge specific - Terraform: batch_job_definition_g5_8xlarge
        job_definitions["g5.8xlarge"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionG58xlarge",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 32,
                "memory": 120000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # G6.4xlarge specific - Terraform: batch_job_definition_g6_4xlarge
        job_definitions["g6.4xlarge"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionG64xlarge",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 16,
                "memory": 60000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # G6.8xlarge specific - Terraform: batch_job_definition_g6_8xlarge
        job_definitions["g6.8xlarge"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionG68xlarge",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 32,
                "memory": 120000,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        # G6e.4xlarge specific - Terraform: batch_job_definition_g6e_4xlarge
        job_definitions["g6e.4xlarge"] = batch.CfnJobDefinition(
            self, "BatchJobDefinitionG6e4xlarge",
            type="container",
            container_properties={
                "image": self.ecr_repo_uri,
                "vcpus": 16,
                "memory": 122880,
                "jobRoleArn": self.task_role.role_arn,
                "command": ["python", "/opt/ml/code/main.py"],
                "privileged": True,
                "environment": [
                    {
                        "name": "LD_LIBRARY_PATH",
                        "value": "/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
                    },
                    {
                        "name": "NVIDIA_DRIVER_CAPABILITIES",
                        "value": "compute,utility,graphics"
                    }
                ],
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "linuxParameters": {
                    "sharedMemorySize": 8192,
                    "devices": [
                        {
                            "hostPath": "/dev/nvidia0",
                            "containerPath": "/dev/nvidia0",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidiactl",
                            "containerPath": "/dev/nvidiactl",
                            "permissions": ["read", "write", "mknod"]
                        },
                        {
                            "hostPath": "/dev/nvidia-uvm",
                            "containerPath": "/dev/nvidia-uvm",
                            "permissions": ["read", "write", "mknod"]
                        }
                    ]
                },
                "mountPoints": [
                    {
                        "sourceVolume": "workspace",
                        "containerPath": "/tmp",
                        "readOnly": False
                    },
                    {
                        "sourceVolume": "shm",
                        "containerPath": "/dev/shm",
                        "readOnly": False
                    }
                ],
                "volumes": [
                    {
                        "name": "workspace",
                        "host": {
                            "sourcePath": "/mnt/workspace"
                        }
                    },
                    {
                        "name": "shm",
                        "host": {
                            "sourcePath": "/dev/shm"
                        }
                    }
                ],
                "ulimits": [
                    {
                        "name": "memlock",
                        "softLimit": -1,
                        "hardLimit": -1
                    },
                    {
                        "name": "stack",
                        "softLimit": 67108864,
                        "hardLimit": 67108864
                    }
                ]
            },
            retry_strategy={
                "attempts": 1
            },
            timeout={
                "attemptDurationSeconds": 259200
            }
        )
        
        return job_definitions