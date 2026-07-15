#!/bin/bash
# Download the trained model from AWS instance.
# Usage: ./download_model.sh

set -e

KEY_NAME="barricade-key"
REGION="us-east-1"

if [ ! -f .training_instance ]; then
    echo "No training instance found."
    exit 1
fi

INSTANCE_ID=$(cat .training_instance)

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Downloading model from $PUBLIC_IP..."

mkdir -p ../checkpoints

scp -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no \
    ubuntu@$PUBLIC_IP:/home/ubuntu/barricade/engine/checkpoints/best_model.pt \
    ../checkpoints/best_model.pt

# Also grab the metrics
scp -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no \
    ubuntu@$PUBLIC_IP:/home/ubuntu/barricade/engine/checkpoints/metrics.json \
    ../checkpoints/metrics.json 2>/dev/null || true

# And TensorBoard logs
scp -r -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no \
    ubuntu@$PUBLIC_IP:/home/ubuntu/barricade/engine/runs/ \
    ../runs/ 2>/dev/null || true

echo ""
echo "Done! Model saved to engine/checkpoints/best_model.pt"
echo "View metrics: tensorboard --logdir engine/runs"
