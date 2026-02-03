#!/bin/bash
# Launch AWS instance to train GPT-2 Large (774M parameters)
# Requires g5.2xlarge (48GB VRAM) or larger

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration for Large model
INSTANCE_TYPE="g5.2xlarge"  # 48GB VRAM
AMI_ID=""
KEY_NAME=""
SECURITY_GROUP=""
REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")
VOLUME_SIZE=150
INSTANCE_NAME="seriguela-large-training"
HF_TOKEN=""
WANDB_KEY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token) HF_TOKEN="$2"; shift 2;;
        --wandb-key) WANDB_KEY="$2"; shift 2;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        --help)
            echo "Usage: $0 --hf-token TOKEN --wandb-key KEY"
            echo "Launches AWS instance to train GPT-2 Large (774M)"
            echo "Default instance: g5.2xlarge (48GB VRAM)"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Validate tokens
if [ -z "$WANDB_KEY" ]; then
    print_error "Wandb API key is required! Use --wandb-key"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    print_warning "HuggingFace token not provided. Model won't be pushed to Hub."
fi

print_status "Launching instance for GPT-2 Large training..."
print_status "Instance type: $INSTANCE_TYPE (48GB VRAM)"

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

# Find key pair
KEY_NAME=$(aws ec2 describe-key-pairs --query "KeyPairs[0].KeyName" --output text 2>/dev/null)
if [ -z "$KEY_NAME" ] || [ "$KEY_NAME" == "None" ]; then
    print_error "No SSH key pair found"
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
    MY_IP=$(curl -s ifconfig.me)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP" \
        --protocol tcp --port 22 \
        --cidr "${MY_IP}/32"
fi
print_status "Using security group: $SECURITY_GROUP"

# Create user-data script for GPT-2 Large training
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x

echo "=========================================="
echo "GPT-2 Large Training Setup"
echo "Started: $(date)"
echo "=========================================="

cloud-init status --wait

sudo -u ubuntu bash << 'UBUNTUSETUP'
cd /home/ubuntu

echo "[1/9] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git

echo "[2/9] Cloning repository..."
git clone https://github.com/augustocsc/seriguela.git
cd seriguela

echo "[3/9] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[4/9] Upgrading pip..."
pip install --upgrade pip -q

echo "[5/9] Installing PyTorch with CUDA..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q

echo "[6/9] Installing requirements..."
pip install -r requirements.txt -q

echo "[7/9] Upgrading Wandb..."
pip install --upgrade 'wandb>=0.24.1' -q

echo "[8/9] Configuring environment..."
export WANDB_API_KEY='WANDB_KEY_PLACEHOLDER'
export HF_TOKEN='HF_TOKEN_PLACEHOLDER'

echo "[9/9] Validating setup..."
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

echo ""
echo "=========================================="
echo "Starting GPT-2 Large Training"
echo "Model: gpt2-large (774M parameters)"
echo "=========================================="

# Start training
cd /home/ubuntu/seriguela
source venv/bin/activate

python3 scripts/train_with_json.py \
  --model_size gpt2-large \
  --dataset_repo augustocsc/sintetico_natural \
  --data_dir 700K \
  --output_dir ./output/gpt2_large_700K_json \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --early_stopping_patience 3 \
  2>&1 | tee /home/ubuntu/training_large.log

echo ""
echo "=========================================="
echo "Training Completed!"
echo "Finished: $(date)"
echo "=========================================="

# Create completion marker
touch /home/ubuntu/.training_complete

# Save results info
cat > /home/ubuntu/training_results.txt << 'RESULTS'
GPT-2 Large Training Completed!

Model: gpt2-large (774M parameters)
Saved to: ~/seriguela/output/gpt2_large_700K_json

Next steps:
1. Test model with REINFORCE:
   cd ~/seriguela
   source venv/bin/activate
   python scripts/debug_reinforce.py \
     --model_path ./output/gpt2_large_700K_json \
     --dataset data/benchmarks/nguyen/nguyen_5.csv \
     --epochs 10

2. Download model to local:
   scp -r ubuntu@IP:~/seriguela/output/gpt2_large_700K_json ./
RESULTS

UBUNTUSETUP
USERDATA
)

# Replace placeholders
USER_DATA="${USER_DATA//WANDB_KEY_PLACEHOLDER/$WANDB_KEY}"
USER_DATA="${USER_DATA//HF_TOKEN_PLACEHOLDER/$HF_TOKEN}"

# Launch instance
print_status "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Model,Value=gpt2-large}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

print_status "Instance launched: $INSTANCE_ID"

# Wait for instance
print_status "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo ""
echo "=========================================="
echo -e "${GREEN}GPT-2 Large Training Instance Ready!${NC}"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Instance Type: $INSTANCE_TYPE (48GB VRAM)"
echo "Public IP: $PUBLIC_IP"
echo ""
echo -e "${BLUE}Monitor training:${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "  tail -f /home/ubuntu/training_large.log"
echo ""
echo -e "${BLUE}Check when complete:${NC}"
echo "  ssh ubuntu@${PUBLIC_IP} 'while [ ! -f ~/.training_complete ]; do sleep 60; echo \"Training in progress...\"; done; cat ~/training_results.txt'"
echo ""
echo -e "${YELLOW}Estimated time:${NC} ~4-5 hours for 3 epochs"
echo -e "${YELLOW}Cost:${NC} ~$8-10 USD ($2/hour for g5.2xlarge)"
echo ""

# Save info
INFO_DIR="${HOME}/.seriguela"
mkdir -p "$INFO_DIR"
cat > "$INFO_DIR/large_instance_info.txt" << INFO
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Key Name: $KEY_NAME
Instance Type: $INSTANCE_TYPE
Model: GPT-2 Large (774M)
Launched: $(date)
INFO

print_status "Instance info saved to: $INFO_DIR/large_instance_info.txt"
