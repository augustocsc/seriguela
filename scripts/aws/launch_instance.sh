#!/bin/bash
# Script to launch and configure AWS g5.xlarge instance for Seriguela training
# Usage: ./launch_instance.sh [--hf-token TOKEN] [--wandb-key KEY]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
            echo "  --hf-token TOKEN     HuggingFace token"
            echo "  --wandb-key KEY      Wandb API key"
            echo "  --instance-type TYPE Instance type (default: g5.xlarge)"
            echo "  --key-name NAME      SSH key pair name"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

print_status "Launching Seriguela training instance..."

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

# Create user-data script for automatic setup
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x

# Wait for cloud-init to complete
cloud-init status --wait

# Setup as ubuntu user
sudo -u ubuntu bash << 'UBUNTUSETUP'
cd /home/ubuntu

# Install dependencies
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git

# Clone repository
git clone https://github.com/augustocsc/seriguela.git
cd seriguela

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip -q
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 -q

# Create marker file to indicate setup complete
touch /home/ubuntu/.setup_complete
UBUNTUSETUP
USERDATA
)

# Add tokens to user-data if provided
if [ -n "$HF_TOKEN" ] || [ -n "$WANDB_KEY" ]; then
    TOKEN_SETUP="
# Configure tokens
cd /home/ubuntu/seriguela
echo 'HF_TOKEN=$HF_TOKEN' > .env
echo 'WANDB_API_KEY=$WANDB_KEY' >> .env
"
    USER_DATA="${USER_DATA}${TOKEN_SETUP}"
fi

# Launch instance
print_status "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
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
echo ""
echo "Connect with:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Check setup progress:"
echo "  ssh ubuntu@${PUBLIC_IP} 'tail -f /var/log/user-data.log'"
echo ""
echo "Wait for setup to complete (check for .setup_complete):"
echo "  ssh ubuntu@${PUBLIC_IP} 'while [ ! -f ~/.setup_complete ]; do sleep 10; done; echo Done!'"
echo ""
echo "Then run training:"
echo "  ssh ubuntu@${PUBLIC_IP} 'cd seriguela && source venv/bin/activate && bash scripts/aws/run_all_training.sh'"
echo ""

# Save instance info
echo "$INSTANCE_ID" > /tmp/seriguela_instance_id.txt
echo "$PUBLIC_IP" > /tmp/seriguela_instance_ip.txt
