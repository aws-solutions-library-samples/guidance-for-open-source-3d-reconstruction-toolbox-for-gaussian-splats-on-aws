from constructs import Construct
from aws_cdk import (
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions
)

class Sns(Construct):
    """Class for SNS Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            admin_email: str,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the SNS topic with default encryption
        self.sns_topic = sns.Topic(  # Changed from self.topic to self.sns_topic
            self,
            "NotificationTopic",
            display_name="3DGS Workflow Notifications"
        )

        # Add email subscription
        self.sns_topic.add_subscription(  # Updated to use sns_topic
            subscriptions.EmailSubscription(admin_email)
        )
