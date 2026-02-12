#!/bin/bash
# Script to launch and configure AWS g5.xlarge instance for Seriguela training
# FIXED VERSION - Includes Wandb validation and proper setup
# Usage: ./launch_instance_fixed.sh [--hf-token TOKEN] [--wandb-key KEY]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default configuration
INSTANCE_TYPE="g5.xlarge"
AMI_ID=""  # Will be auto-detected
KEY_NAME=""  # Will be auto-detected
SECURITY_GROUP=""  # Will be auto-detected or created
REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")
VOLUME_SIZE=100
INSTANCE_NAME="augusto-seriguela-training"
HF_TOKEN=""
WANDB_KEY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token) HF_TOKEN="$2"; shift 2;;
        --wandb-key) WANDB_KEY="$2"; shift 2;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        --key-name) KEY_NAME="$2"; shift 2;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --hf-token TOKEN     HuggingFace token (required for push to hub)"
            echo "  --wandb-key KEY      Wandb API key (required for logging)"
            echo "  --instance-type TYPE Instance type (default: g5.xlarge)"
            echo "  --key-name NAME      SSH key pair name"
            echo ""
            echo "Example:"
            echo "  $0 --hf-token hf_xxx --wandb-key wandb_v1_xxx"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Validate required tokens
if [ -z "$WANDB_KEY" ]; then
    print_error "Wandb API key is required! Use --wandb-key"
    print_warning "Get your key from: https://wandb.ai/authorize"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    print_warning "HuggingFace token not provided. Model won't be pushed to Hub."
    print_warning "Get your token from: https://huggingface.co/settings/tokens"
fi

print_status "Launching Seriguela training instance with validated setup..."

# Find Deep Learning AMI
print_status "Finding Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
    --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    print_error "Could not find Deep Learning AMI"
    exit 1
fi
print_status "Using AMI: $AMI_ID"

# Find or select key pair
if [ -z "$KEY_NAME" ]; then
    KEY_NAME=$(aws ec2 describe-key-pairs --query "KeyPairs[0].KeyName" --output text 2>/dev/null)
fi
if [ -z "$KEY_NAME" ] || [ "$KEY_NAME" == "None" ]; then
    print_error "No SSH key pair found. Create one first or specify with --key-name"
    exit 1
fi
print_status "Using key pair: $KEY_NAME"

# Find or create security group
SECURITY_GROUP=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=seriguela-sg" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null)

if [ -z "$SECURITY_GROUP" ] || [ "$SECURITY_GROUP" == "None" ]; then
    print_status "Creating security group..."
    SECURITY_GROUP=$(aws ec2 create-security-group \
        --group-name seriguela-sg \
        --description "Security group for Seriguela training" \
        --query "GroupId" --output text)

    # Get current IP and add SSH rule
    MY_IP=$(curl -s ifconfig.me)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP" \
        --protocol tcp --port 22 \
        --cidr "${MY_IP}/32"
    print_status "Created security group with SSH access from $MY_IP"
else
    # Update security group with current IP
    MY_IP=$(curl -s ifconfig.me)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP" \
        --protocol tcp --port 22 \
        --cidr "${MY_IP}/32" 2>/dev/null || true
fi
print_status "Using security group: $SECURITY_GROUP"

# Create user-data script for automatic setup with validation
USER_DATA=$(cat << USERDATA
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x

echo "=========================================="
echo "Seriguela Instance Setup - VALIDATED"
echo "Started: \$(date)"
echo "=========================================="

# Wait for cloud-init to complete
cloud-init status --wait

# Setup as ubuntu user
sudo -u ubuntu bash << 'UBUNTUSETUP'
cd /home/ubuntu

echo "[1/8] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git dos2unix

echo "[2/8] Cloning repository..."
git clone https://github.com/augustocsc/seriguela.git
cd seriguela

echo "[3/8] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[4/8] Upgrading pip..."
pip install --upgrade pip -q

echo "[5/8] Installing requirements..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 -q

echo "[6/8] Upgrading Wandb to latest version..."
pip install --upgrade 'wandb>=0.24.1' -q

echo "[7/8] Configuring environment..."
# Create .env file
cat > .env << 'ENVFILE'
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_KEY
ENVFILE

echo "[8/8] Validating setup..."

# Validate Python packages
python3 << 'PYCHECK'
import sys
print("Testing imports...")
try:
    import transformers
    print(f"✅ transformers {transformers.__version__}")
    import torch
    print(f"✅ torch {torch.__version__}")
    import wandb
    print(f"✅ wandb {wandb.__version__}")
    import peft
    print(f"✅ peft {peft.__version__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
PYCHECK

if [ \$? -ne 0 ]; then
    echo "❌ Package validation failed"
    exit 1
fi

# Validate GPU
echo "Checking GPU..."
if nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "❌ No GPU detected"
    exit 1
fi

# Validate Wandb authentication
if [ -n "$WANDB_KEY" ]; then
    echo "Validating Wandb authentication..."
    python3 << PYVALIDATE
import wandb
import os
try:
    result = wandb.login(key='$WANDB_KEY')
    if result:
        print("✅ Wandb authentication successful")
        # Get user info
        import requests
        response = requests.get('https://api.wandb.ai/graphql',
                              headers={'Authorization': f'Bearer $WANDB_KEY'},
                              json={'query': '{viewer{entity}}'})
        if response.status_code == 200:
            print(f"   Logged in to Wandb")
    else:
        print("❌ Wandb authentication failed")
        exit(1)
except Exception as e:
    print(f"❌ Wandb validation error: {e}")
    exit(1)
PYVALIDATE

    if [ \$? -ne 0 ]; then
        echo "❌ Wandb authentication failed"
        exit 1
    fi
else
    echo "⚠️  No Wandb key provided - skipping validation"
fi

# Validate HuggingFace token
if [ -n "$HF_TOKEN" ]; then
    echo "Validating HuggingFace authentication..."
    python3 << PYVALIDATE
from huggingface_hub import HfApi
try:
    api = HfApi(token='$HF_TOKEN')
    user = api.whoami()
    print(f"✅ HuggingFace authentication successful")
    print(f"   Logged in as: {user.get('name', 'unknown')}")
except Exception as e:
    print(f"❌ HuggingFace validation error: {e}")
    exit(1)
PYVALIDATE

    if [ \$? -ne 0 ]; then
        echo "❌ HuggingFace authentication failed"
        exit 1
    fi
else
    echo "⚠️  No HuggingFace token provided - model won't be pushed to Hub"
fi

# All validations passed
echo ""
echo "=========================================="
echo "✅ Setup Complete and Validated!"
echo "Finished: \$(date)"
echo "=========================================="

# Create completion markers
touch /home/ubuntu/.setup_complete
touch /home/ubuntu/.setup_validated

# Create info file
cat > /home/ubuntu/setup_info.txt << 'INFOFILE'
Setup completed successfully!

Validated:
- Python packages installed
- GPU detected
- Wandb authenticated
- HuggingFace authenticated (if token provided)

Ready to train!

Quick commands:
  cd ~/seriguela
  source venv/bin/activate
  python scripts/train.py --help

Monitor scripts:
  bash scripts/aws/monitor_training_auto.sh
INFOFILE

echo "Setup info saved to ~/setup_info.txt"
UBUNTUSETUP

# End of setup
echo "User-data script completed"
USERDATA
)

# Replace placeholder tokens in user-data
USER_DATA="${USER_DATA//\$HF_TOKEN/$HF_TOKEN}"
USER_DATA="${USER_DATA//\$WANDB_KEY/$WANDB_KEY}"

# Launch instance
print_status "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=seriguela},{Key=AutoSetup,Value=validated}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

print_status "Instance launched: $INSTANCE_ID"

# Wait for instance to be running
print_status "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo ""
echo "=========================================="
echo -e "${GREEN}Instance Ready!${NC}"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Key Pair: $KEY_NAME"
echo ""
echo -e "${BLUE}Connect with:${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo -e "${BLUE}Check setup progress:${NC}"
echo "  ssh ubuntu@${PUBLIC_IP} 'tail -f /var/log/user-data.log'"
echo ""
echo -e "${BLUE}Wait for VALIDATED setup to complete:${NC}"
echo "  ssh ubuntu@${PUBLIC_IP} 'while [ ! -f ~/.setup_validated ]; do sleep 10; echo \"Setup in progress...\"; done; echo \"✅ Setup validated!\"; cat ~/setup_info.txt'"
echo ""
echo -e "${BLUE}Then run training:${NC}"
echo "  ssh ubuntu@${PUBLIC_IP} 'cd seriguela && source venv/bin/activate && bash scripts/aws/run_all_training.sh'"
echo ""
echo -e "${YELLOW}Setup includes:${NC}"
echo "  ✅ Wandb 0.24.1+ with authentication test"
echo "  ✅ HuggingFace authentication test"
echo "  ✅ GPU validation"
echo "  ✅ All packages validated"
echo ""

# Save instance info
INFO_DIR="${HOME}/.seriguela"
mkdir -p "$INFO_DIR"
echo "$INSTANCE_ID" > "$INFO_DIR/last_instance_id.txt"
echo "$PUBLIC_IP" > "$INFO_DIR/last_instance_ip.txt"
echo "$KEY_NAME" > "$INFO_DIR/last_key_name.txt"

cat > "$INFO_DIR/last_instance_info.txt" << INFOEND
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Key Name: $KEY_NAME
Instance Type: $INSTANCE_TYPE
Region: $REGION
Launched: $(date)
Setup: Validated (Wandb + HF + GPU)
INFOEND

print_status "Instance info saved to: $INFO_DIR/"
echo ""
