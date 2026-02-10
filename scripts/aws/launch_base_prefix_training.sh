#!/bin/bash
# Launch GPT-2 Base training with prefix dataset on AWS
# Usage: ./launch_base_prefix_training.sh --hf-token <token> --wandb-key <key>

set -e

# Parse arguments
HF_TOKEN=""
WANDB_KEY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --wandb-key) WANDB_KEY="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
  echo "Usage: $0 --hf-token <token> --wandb-key <key>"
  exit 1
fi

# Instance configuration
INSTANCE_TYPE="g5.xlarge"
IMAGE_ID="ami-01dfe92df9055a1c6"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04)
KEY_NAME="chave-gpu-nova"
SECURITY_GROUP="sg-0deaa73e23482e3f6"
VOLUME_SIZE=100

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_NAME="seriguela-base-prefix-training-${TIMESTAMP}"

echo "============================================"
echo "Launching instance: $INSTANCE_NAME"
echo "Instance type: $INSTANCE_TYPE"
echo "Volume: ${VOLUME_SIZE}GB"
echo "============================================"

# Create user data script
TEMP_DIR="C:/Users/madeinweb/AppData/Local/Temp"
mkdir -p "$TEMP_DIR" 2>/dev/null
cat > "$TEMP_DIR/userdata_base_prefix.sh" <<'EOF'
#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

sleep 5

# Clone repository
cd /home/ubuntu
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
git checkout experiment/ppo-symbolic-regression

# Install Python dependencies (PyTorch already installed in AMI)
pip3 install -r requirements.txt

# Setup credentials
echo "huggingface = ${HF_TOKEN}" > /home/ubuntu/.tokens.txt
echo "wandb = ${WANDB_KEY}" >> /home/ubuntu/.tokens.txt
chmod 600 /home/ubuntu/.tokens.txt

# Login to services
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_KEY}"
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_KEY

# Start training with FIXED script (no double-split)
python3 scripts/train_with_json_fixed.py \
  --model_size gpt2 \
  --dataset_repo augustocsc/sintetico_natural_prefix_682k \
  --text_column p_prompt_n_converted \
  --output_dir ./output/gpt2_base_prefix_682k \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --lora_r 8 \
  --lora_alpha 32 \
  --early_stopping_patience 3 \
  > /home/ubuntu/training_base_prefix.log 2>&1

# Mark completion
touch /home/ubuntu/.training_complete
echo "Training complete at $(date)" >> /home/ubuntu/.training_complete

# Upload model to HuggingFace
cd output/gpt2_base_prefix_682k
huggingface-cli repo create gpt2_base_prefix_682k --type model || true
huggingface-cli upload gpt2_base_prefix_682k . . --commit-message "GPT-2 Base trained on prefix dataset (682K)"

EOF

# Replace tokens in user data
sed -i "s|\${HF_TOKEN}|${HF_TOKEN}|g" $TEMP_DIR/userdata_base_prefix.sh
sed -i "s|\${WANDB_KEY}|${WANDB_KEY}|g" $TEMP_DIR/userdata_base_prefix.sh

# Launch instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $IMAGE_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\"}}]" \
  --user-data file://$TEMP_DIR/userdata_base_prefix.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to start..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo ""
echo "============================================"
echo "Instance ready!"
echo "============================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "To monitor training:"
echo "  ssh -i ~/.ssh/chave-gpu.pem ubuntu@${PUBLIC_IP}"
echo "  tail -f ~/training_base_prefix.log"
echo ""
echo "To check completion:"
echo "  ssh ubuntu@${PUBLIC_IP} 'test -f ~/.training_complete && echo DONE || echo Running'"
echo ""
echo "To download model after training:"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ubuntu@${PUBLIC_IP}:~/seriguela/output/gpt2_base_prefix_682k ./"
echo ""
echo "IMPORTANT: Stop instance when done!"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo "============================================"

rm $TEMP_DIR/userdata_base_prefix.sh
