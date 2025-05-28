from aws_cdk import (
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_iam as iam,
    Duration,
    aws_s3_notifications,
    Environment,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct

class S3(Construct):
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            bucket_name: str,
            trigger_lambda_function: lambda_.Function,
            s3_trigger_key: str,
            s3_trigger_extension: str,
            maintain_s3_objects_on_stack_deletion: str,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        removal_policy = RemovalPolicy.DESTROY
        if maintain_s3_objects_on_stack_deletion.lower() == "true":
            removal_policy = RemovalPolicy.RETAIN
            
        # Create a logging bucket with proper configuration
        self.log_bucket = s3.Bucket(
            self,
            "LoggingBucket",
            bucket_name=f"{bucket_name}-logs",
            versioned=True,
            enforce_ssl=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=removal_policy,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=Duration.days(90)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(180)
                        )
                    ],
                    expiration=Duration.days(365)
                )
            ]
        )

        # Grant S3 log delivery permissions to the logging bucket
        self.log_bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="S3ServerAccessLogsPolicy",
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal("logging.s3.amazonaws.com")],
                actions=["s3:PutObject"],
                resources=[f"{self.log_bucket.bucket_arn}/*"],
                conditions={
                    "StringEquals": {
                        "aws:SourceAccount": env.account
                    },
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:s3:::{bucket_name}"
                    }
                }
            )
        )

        # Create the main bucket with proper configuration
        self.bucket = s3.Bucket(
            self,
            "AssetBucket",
            bucket_name=bucket_name,
            versioned=True,
            enforce_ssl=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
            cors=[
                s3.CorsRule(
                    allowed_methods=[
                        s3.HttpMethods.GET,
                        s3.HttpMethods.POST,
                        s3.HttpMethods.PUT
                    ],
                    allowed_origins=["*"],
                    allowed_headers=["*"],
                    max_age=3000
                )
            ],
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="access-logs/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="TransitionRule",
                    enabled=True,
                    prefix="documents/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                ),
                s3.LifecycleRule(
                    id="CleanupOldVersions",
                    enabled=True,
                    noncurrent_version_transitions=[
                        s3.NoncurrentVersionTransition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(7)
                        )
                    ],
                    noncurrent_version_expiration=Duration.days(90)
                ),
                s3.LifecycleRule(
                    id="AbortIncompleteUploads",
                    enabled=True,
                    abort_incomplete_multipart_upload_after=Duration.days(7)
                )
            ]
        )

        # Add secure transport policy
        self.bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="DenyNonSecureTransport",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["s3:*"],
                resources=[
                    self.bucket.bucket_arn,
                    f"{self.bucket.bucket_arn}/*"
                ],
                conditions={
                    "Bool": {
                        "aws:SecureTransport": "false"
                    }
                }
            )
        )

        # Grant Lambda permissions
        self.bucket.grant_read_write(trigger_lambda_function)

        # Add S3 notification
        try:
            notification = aws_s3_notifications.LambdaDestination(trigger_lambda_function)
            
            self.bucket.add_event_notification(
                s3.EventType.OBJECT_CREATED_PUT,
                notification,
                s3.NotificationKeyFilter(
                    prefix=s3_trigger_key,
                    suffix=s3_trigger_extension
                )
            )
        except Exception as e:
            print(f"Error setting up S3 notification: {str(e)}")

        # Add outputs
        CfnOutput(
            self,
            "BucketName",
            value=self.bucket.bucket_name,
            description="Name of the created S3 bucket"
        )

        CfnOutput(
            self,
            "LoggingBucketName",
            value=self.log_bucket.bucket_name,
            description="Name of the logging S3 bucket"
        )
