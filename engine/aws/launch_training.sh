#!/bin/bash
# Launch a GPU instance on AWS and start AlphaZero training.
# Prerequisites: aws cli configured, a key pair created, and a security group allowing SSH.
#
# Usage:
#   chmod +x launch_training.sh
#   ./launch_training.sh
#
# To check progress:
#   ssh -i ~/.ssh/barricade-key.pem ubuntu@<instance-ip> "tail -f /home/ubuntu/barricade/train.log"
#
# To stop:
#   aws ec2 terminate-instances --instance-ids <id>

set -e

# ============ CONFIG ============
INSTANCE_TYPE="g5.xlarge"           # A10G GPU, ~$1/hr
AMI="ami-072e487908654a0d2"        # Deep Learning Base OSS Nvidia, Ubuntu 22.04 (us-east-1)
KEY_NAME="vla-key-pair"             # Your SSH key pair name
SECURITY_GROUP="barricade-sg"       # Security group allowing SSH (port 22)
REGION="us-east-1"
VOLUME_SIZE=100                     # GB
KEY_PATH="D:/Downloads/vla-key-pair.pem"

# Training config
ITERATIONS=50
SELF_PLAY_GAMES=200
SIMULATIONS=400
ARENA_GAMES=40
EPOCHS=10
BATCH_SIZE=64
LR=0.001

echo "=== Barricade AI Training - AWS Launch Script ==="
echo "Instance: $INSTANCE_TYPE | Region: $REGION"
echo "Training: $ITERATIONS iterations × $SELF_PLAY_GAMES games × $SIMULATIONS sims"
echo ""

# ============ CREATE KEY PAIR (if needed) ============
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" 2>/dev/null | grep -q "$KEY_NAME"; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo "Key saved to ~/.ssh/${KEY_NAME}.pem"
fi

# ============ USER DATA (runs on instance boot) ============
USER_DATA=$(cat <<'SETUP'
#!/bin/bash
set -e
exec > /var/log/barricade-setup.log 2>&1

cd /home/ubuntu

# Install dependencies
apt-get update -y
apt-get install -y python3-pip git

# Clone repo (or copy — we'll use git)
git clone https://github.com/KeshavM05/Quoridor.git barricade || true
cd barricade/engine

# Install Python deps with CUDA PyTorch
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements.txt
pip3 install tensorboard

# Ensure training_runs directory exists and persists on the EBS volume
mkdir -p /home/ubuntu/barricade/engine/training_runs
mkdir -p /home/ubuntu/barricade/engine/checkpoints

# Start training
nohup python3 train.py \
    --iterations ITER_PLACEHOLDER \
    --self-play-games SPG_PLACEHOLDER \
    --simulations SIM_PLACEHOLDER \
    --arena-games AG_PLACEHOLDER \
    --epochs EP_PLACEHOLDER \
    --batch-size BS_PLACEHOLDER \
    --lr LR_PLACEHOLDER \
    --device cuda \
    --checkpoint-dir /home/ubuntu/barricade/engine/checkpoints \
    > /home/ubuntu/barricade/train.log 2>&1 &

echo "Training started at $(date)" > /home/ubuntu/barricade/status.txt
echo "Training runs will be saved to /home/ubuntu/barricade/engine/training_runs/" >> /home/ubuntu/barricade/status.txt
SETUP
)

# Replace placeholders
USER_DATA="${USER_DATA//ITER_PLACEHOLDER/$ITERATIONS}"
USER_DATA="${USER_DATA//SPG_PLACEHOLDER/$SELF_PLAY_GAMES}"
USER_DATA="${USER_DATA//SIM_PLACEHOLDER/$SIMULATIONS}"
USER_DATA="${USER_DATA//AG_PLACEHOLDER/$ARENA_GAMES}"
USER_DATA="${USER_DATA//EP_PLACEHOLDER/$EPOCHS}"
USER_DATA="${USER_DATA//BS_PLACEHOLDER/$BATCH_SIZE}"
USER_DATA="${USER_DATA//LR_PLACEHOLDER/$LR}"

# ============ LAUNCH INSTANCE ============
echo "Launching $INSTANCE_TYPE instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --region "$REGION" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=barricade-training}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to start..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== INSTANCE RUNNING ==="
echo "IP:          $PUBLIC_IP"
echo "Instance ID: $INSTANCE_ID"
echo ""
echo "=== USEFUL COMMANDS ==="
echo "SSH in:      ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "Watch logs:  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'tail -f /home/ubuntu/barricade/train.log'"
echo "TensorBoard: ssh -i ~/.ssh/${KEY_NAME}.pem -L 6006:localhost:6006 ubuntu@$PUBLIC_IP 'tensorboard --logdir /home/ubuntu/barricade/engine/runs'"
echo "Copy model:  scp -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP:/home/ubuntu/barricade/engine/checkpoints/best_model.pt ./engine/checkpoints/"
echo "Terminate:   aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "=== SSH PORT FORWARDING (for dashboard/TensorBoard access) ==="
echo "Forward TensorBoard:  ssh -i ~/.ssh/${KEY_NAME}.pem -N -L 6006:localhost:6006 ubuntu@$PUBLIC_IP"
echo "Forward FastAPI:      ssh -i ~/.ssh/${KEY_NAME}.pem -N -L 8000:localhost:8000 ubuntu@$PUBLIC_IP"
echo "Forward both:         ssh -i ~/.ssh/${KEY_NAME}.pem -N -L 6006:localhost:6006 -L 8000:localhost:8000 ubuntu@$PUBLIC_IP"
echo ""
echo "=== RETRIEVE TRAINING JOURNAL ==="
echo "Copy full run:    scp -r -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP:/home/ubuntu/barricade/engine/training_runs/ ./engine/training_runs/"
echo "Copy latest run:  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'ls -td /home/ubuntu/barricade/engine/training_runs/run_*/ | head -1' | xargs -I{} scp -r -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP:{} ./engine/training_runs/"
echo "View summary:     ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'cat /home/ubuntu/barricade/engine/training_runs/run_*/summary.md 2>/dev/null || echo Training still in progress'"
echo ""
echo "Training will start automatically (~2 min for setup)."
echo "Estimated cost: ~\$35-50 for full run (${INSTANCE_TYPE} @ ~\$1/hr × ~35-50hrs)"
echo ""
echo "NOTE: training_runs/ directory is preserved on the EBS volume."
echo "Make sure to copy it before terminating the instance!"
echo ""
echo "Instance ID saved to .training_instance"
echo "$INSTANCE_ID" > .training_instance
