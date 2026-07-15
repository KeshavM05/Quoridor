#!/bin/bash
# Check training progress on the running AWS instance.
# Usage: ./check_training.sh

set -e

KEY_NAME="barricade-key"
REGION="us-east-1"

if [ ! -f .training_instance ]; then
    echo "No training instance found. Run launch_training.sh first."
    exit 1
fi

INSTANCE_ID=$(cat .training_instance)

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text 2>/dev/null)

if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
    echo "Instance $INSTANCE_ID is not running."
    exit 1
fi

echo "=== Training Status ==="
echo "Instance: $INSTANCE_ID ($PUBLIC_IP)"
echo ""

ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'EOF'
echo "--- Last 20 lines of training log ---"
tail -20 /home/ubuntu/barricade/train.log 2>/dev/null || echo "(no log yet — setup still running)"
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not ready)"
echo ""
echo "--- Checkpoints ---"
ls -la /home/ubuntu/barricade/engine/checkpoints/ 2>/dev/null || echo "(no checkpoints yet)"
EOF
