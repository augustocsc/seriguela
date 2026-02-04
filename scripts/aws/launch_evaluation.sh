#!/bin/bash
# Launch AWS instance for model evaluation with GPU

set -e

# Default values
INSTANCE_TYPE="g5.xlarge"
VOLUME_SIZE=100
INSTANCE_NAME="seriguela-evaluation"
KEY_NAME="chave-gpu-nova"
SECURITY_GROUP="seriguela-sg"
AMI_ID=""  # Will be auto-detected

# Parse arguments
HF_TOKEN=""
WANDB_KEY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --wandb-key)
      WANDB_KEY="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
  echo "Error: --hf-token and --wandb-key are required"
  exit 1
fi

echo "Launching evaluation instance..."
echo "Instance type: $INSTANCE_TYPE"

# Create user data script
cat > aws/temp/userdata_eval.sh << 'USERDATA_EOF'
#!/bin/bash
set -x
exec > >(tee /home/ubuntu/setup.log) 2>&1

echo "=== Starting evaluation setup ==="
date

# Update system
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv

# Setup as ubuntu user
cd /home/ubuntu

# Clone repository
if [ ! -d "seriguela" ]; then
    sudo -u ubuntu git clone https://github.com/augustocsc/seriguela.git
    cd seriguela
else
    cd seriguela
    sudo -u ubuntu git pull
fi

# Create virtual environment
sudo -u ubuntu python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Set tokens
echo "export HF_TOKEN='PLACEHOLDER_HF_TOKEN'" >> ~/.bashrc
echo "export WANDB_API_KEY='PLACEHOLDER_WANDB_KEY'" >> ~/.bashrc
source ~/.bashrc

# Create results directory
mkdir -p results

# Mark setup complete
touch /home/ubuntu/.evaluation_ready

echo "=== Evaluation setup complete ==="
date
USERDATA_EOF

# Replace placeholders in userdata
sed -i "s/PLACEHOLDER_HF_TOKEN/$HF_TOKEN/g" aws/temp/userdata_eval.sh
sed -i "s/PLACEHOLDER_WANDB_KEY/$WANDB_KEY/g" aws/temp/userdata_eval.sh

# Auto-detect AMI if not set
if [ -z "$AMI_ID" ]; then
    echo "Auto-detecting Deep Learning AMI..."
    AMI_ID=$(aws ec2 describe-images \
      --owners amazon \
      --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
      "Name=state,Values=available" \
      --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' \
      --output text \
      --region us-east-1)

    if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
        echo "ERROR: Could not find AMI"
        exit 1
    fi
    echo "Using AMI: $AMI_ID"
fi

# Launch instance
echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
  --user-data file://aws/temp/userdata_eval.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo ""
echo "==================================="
echo "Evaluation instance ready!"
echo "==================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Waiting 60 seconds for system to initialize..."
sleep 60
echo ""
echo "To check setup progress:"
echo "  ssh -i ~/.ssh/chave-gpu.pem ubuntu@$PUBLIC_IP 'tail -f ~/setup.log'"
echo ""
echo "To upload models:"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ./output/gpt2_base_700K_json ubuntu@$PUBLIC_IP:~/seriguela/output/"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ./output/gpt2_medium_700K_json ubuntu@$PUBLIC_IP:~/seriguela/output/"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ./output/gpt2_large_700K_json ubuntu@$PUBLIC_IP:~/seriguela/output/"
echo ""
echo "Instance info saved to: aws/evaluation_instance.txt"

# Save instance info
mkdir -p aws
cat > aws/evaluation_instance.txt << EOF
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Instance Type: $INSTANCE_TYPE
Launch Time: $(date)
EOF

echo "Done!"
