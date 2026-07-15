#!/bin/bash
# Terminate the training instance to stop billing.
# Usage: ./stop_training.sh

set -e

REGION="us-east-1"

if [ ! -f .training_instance ]; then
    echo "No training instance found."
    exit 1
fi

INSTANCE_ID=$(cat .training_instance)

echo "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Instance terminated. Billing stopped."
rm -f .training_instance
