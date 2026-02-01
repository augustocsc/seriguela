#!/bin/bash
# Script to launch AWS instance for model evaluation
# Evaluates two models: original (Se124M_700K_infix) vs v2 (with end token)
# Usage: ./launch_evaluation_instance.sh [--hf-token TOKEN]

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
AMI_ID=""
KEY_NAME=""
SECURITY_GROUP=""
REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")
VOLUME_SIZE=80
INSTANCE_NAME="seriguela-evaluation"
HF_TOKEN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token) HF_TOKEN="$2"; shift 2;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        --key-name) KEY_NAME="$2"; shift 2;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --hf-token TOKEN     HuggingFace token (optional, for accessing models)"
            echo "  --instance-type TYPE Instance type (default: g5.xlarge)"
            echo "  --key-name NAME      SSH key pair name"
            echo ""
            echo "Example:"
            echo "  $0 --hf-token hf_xxx"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$HF_TOKEN" ]; then
    print_warning "HuggingFace token not provided. Public models will still work."
    print_warning "Get your token from: https://huggingface.co/settings/tokens"
fi

print_status "Launching Seriguela evaluation instance..."

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
        --description "Security group for Seriguela" \
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

# Create user-data script for automatic setup
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x

echo "=========================================="
echo "Seriguela Evaluation Instance Setup"
echo "Started: $(date)"
echo "=========================================="

# Wait for cloud-init to complete
cloud-init status --wait

# Setup as ubuntu user
sudo -u ubuntu bash << 'UBUNTUSETUP'
cd /home/ubuntu

echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git jq

echo "[2/7] Cloning repository..."
git clone https://github.com/augustocsc/seriguela.git
cd seriguela

echo "[3/7] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[4/7] Upgrading pip..."
pip install --upgrade pip -q

echo "[5/7] Installing requirements..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 -q

echo "[6/7] Testing setup..."
python3 << 'PYCHECK'
import sys
print("Testing imports...")
try:
    import transformers
    print(f"✅ transformers {transformers.__version__}")
    import torch
    print(f"✅ torch {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    import peft
    print(f"✅ peft {peft.__version__}")
    import datasets
    print(f"✅ datasets {datasets.__version__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo "❌ Package validation failed"
    exit 1
fi

echo "[7/7] Checking GPU..."
if nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected (will be slower)"
fi

# Configure HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
    echo "Configuring HuggingFace authentication..."
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    echo "✅ HuggingFace token configured"
fi

# Make evaluation script executable
chmod +x ~/seriguela/scripts/aws/evaluate_models.sh

# Create completion marker
touch /home/ubuntu/.setup_complete

# Create info file
cat > /home/ubuntu/setup_info.txt << 'INFOFILE'
Seriguela Evaluation Instance - Ready!

Setup completed successfully:
- Python packages installed
- GPU available (if supported)
- Repository cloned and configured

To run the evaluation:
  cd ~/seriguela
  source venv/bin/activate
  bash scripts/aws/evaluate_models.sh

This will compare:
  - Model 1: augustocsc/Se124M_700K_infix (original)
  - Model 2: augustocsc/Se124M_700K_infix_v2 (with <|endofex|> token)

On 500 test samples to evaluate if the ending token improves generation stopping.
INFOFILE

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "Finished: $(date)"
echo "=========================================="
cat ~/setup_info.txt

UBUNTUSETUP

echo "User-data script completed"
USERDATA
)

# Replace HF_TOKEN placeholder
USER_DATA="${USER_DATA//\$HF_TOKEN/$HF_TOKEN}"

# Launch instance
print_status "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=seriguela},{Key=Purpose,Value=evaluation}]" \
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
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/user-data.log'"
echo ""
echo -e "${BLUE}Wait for setup to complete (takes ~5-10 minutes):${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'while [ ! -f ~/.setup_complete ]; do sleep 10; echo \"Setup in progress...\"; done; echo \"✅ Setup complete!\"; cat ~/setup_info.txt'"
echo ""
echo -e "${BLUE}Then run evaluation:${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'cd seriguela && source venv/bin/activate && bash scripts/aws/evaluate_models.sh'"
echo ""
echo -e "${BLUE}Or run in one command:${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'cd seriguela && source venv/bin/activate && nohup bash scripts/aws/evaluate_models.sh > evaluation.log 2>&1 &'"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC} Remember to stop the instance when done:"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo ""

# Save instance info
INFO_DIR="${HOME}/.seriguela"
mkdir -p "$INFO_DIR"
echo "$INSTANCE_ID" > "$INFO_DIR/last_evaluation_instance_id.txt"
echo "$PUBLIC_IP" > "$INFO_DIR/last_evaluation_instance_ip.txt"
echo "$KEY_NAME" > "$INFO_DIR/last_evaluation_key_name.txt"

cat > "$INFO_DIR/last_evaluation_instance_info.txt" << INFOEND
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Key Name: $KEY_NAME
Instance Type: $INSTANCE_TYPE
Region: $REGION
Launched: $(date)
Purpose: Model Evaluation (v1 vs v2)
INFOEND

print_status "Instance info saved to: $INFO_DIR/"
echo ""
