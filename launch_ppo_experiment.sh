#!/bin/bash
# =============================================================================
# Launch PPO Symbolic Regression Experiment on AWS
# =============================================================================
# Usage:
#   ./launch_ppo_experiment.sh --hf-token YOUR_HF_TOKEN --wandb-key YOUR_WANDB_KEY
# =============================================================================

set -e

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hf-token) HF_TOKEN="$2"; shift ;;
        --wandb-key) WANDB_KEY="$2"; shift ;;
        --instance-type) INSTANCE_TYPE="$2"; shift ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Verify required tokens
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: --hf-token is required"
    echo "Usage: ./launch_ppo_experiment.sh --hf-token YOUR_HF_TOKEN --wandb-key YOUR_WANDB_KEY"
    exit 1
fi

if [ -z "$WANDB_KEY" ]; then
    echo "WARNING: --wandb-key not provided. W&B logging will be disabled."
    WANDB_KEY="disabled"
fi

# Configuration
REGION="us-east-1"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
KEY_NAME="chave-gpu"
SECURITY_GROUP="sg-0deaa73e23482e3f6"
INSTANCE_NAME="seriguela-ppo-experiment"

# Ubuntu Deep Learning AMI with PyTorch 2.0 and CUDA 12.1
# This AMI has NVIDIA drivers pre-installed
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS (adjust based on region)

# For g5 instances, use Deep Learning AMI
# Search: aws ec2 describe-images --filters "Name=name,Values=*Deep Learning AMI*Ubuntu*" --query 'Images[*].[ImageId,Name]' --output table
# Or use generic Ubuntu and let userdata install drivers

echo "=========================================="
echo "PPO Symbolic Regression Experiment Launch"
echo "=========================================="
echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE"
echo "AMI: $AMI_ID"
echo "=========================================="

# Check if key exists
if [ ! -f "aws/keys/chave-gpu.pem" ] && [ ! -f "$HOME/.ssh/chave-gpu.pem" ]; then
    echo "WARNING: SSH key not found locally."
    echo "Make sure you have the key to connect to the instance."
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI not found. Please install it first."
    exit 1
fi

# Prepare userdata with tokens
USERDATA_TEMPLATE="userdata_ppo_experiment.sh"
USERDATA_FILE="/tmp/userdata_ppo_experiment_$$.sh"

if [ ! -f "$USERDATA_TEMPLATE" ]; then
    echo "ERROR: Userdata template not found: $USERDATA_TEMPLATE"
    exit 1
fi

# Substitute tokens in userdata
sed -e "s|__HF_TOKEN__|$HF_TOKEN|g" \
    -e "s|__WANDB_KEY__|$WANDB_KEY|g" \
    "$USERDATA_TEMPLATE" > "$USERDATA_FILE"

echo "Userdata prepared with tokens."

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN - Userdata content:"
    echo "=========================================="
    head -60 "$USERDATA_FILE"
    echo "..."
    echo "=========================================="
    rm -f "$USERDATA_FILE"
    exit 0
fi

echo "Launching instance..."

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
    --user-data file://"$USERDATA_FILE" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"

# Clean up temp file
rm -f "$USERDATA_FILE"

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=========================================="
echo "INSTANCE LAUNCHED SUCCESSFULLY"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "=========================================="
echo ""
echo "Connect with:"
echo "  ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Check setup progress:"
echo "  ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP 'tail -50 /var/log/user-data.log'"
echo ""
echo "Check experiment progress:"
echo "  ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP 'tail -50 /home/ubuntu/ppo_experiment.log'"
echo ""
echo "Check if complete:"
echo "  ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP 'ls -la /home/ubuntu/.ppo_experiment_complete'"
echo ""
echo "Stop instance when done:"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo ""
echo "Estimated cost: ~\$1/hour. Remember to stop the instance!"
echo "=========================================="

# Save instance info to file
cat > "PPO_INSTANCE_INFO.md" << EOF
# PPO Experiment Instance

**Launched:** $(date)

| Property | Value |
|----------|-------|
| Instance ID | $INSTANCE_ID |
| Public IP | $PUBLIC_IP |
| Instance Type | $INSTANCE_TYPE |
| Region | $REGION |

## SSH Commands

\`\`\`bash
# Connect
ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP

# Check setup log
ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP 'tail -100 /var/log/user-data.log'

# Check experiment log
ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP 'tail -100 /home/ubuntu/ppo_experiment.log'

# Check if complete
ssh -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP 'ls -la /home/ubuntu/.ppo_experiment_complete'
\`\`\`

## Download Results

\`\`\`bash
# After experiment completes
scp -i aws/keys/chave-gpu.pem -r ubuntu@$PUBLIC_IP:/home/ubuntu/seriguela/output/ppo_experiments ./results/
scp -i aws/keys/chave-gpu.pem ubuntu@$PUBLIC_IP:/home/ubuntu/*.log ./logs/
\`\`\`

## Stop Instance

\`\`\`bash
aws ec2 stop-instances --instance-ids $INSTANCE_ID
\`\`\`
EOF

echo "Instance info saved to PPO_INSTANCE_INFO.md"
