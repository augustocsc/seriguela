#!/bin/bash
#
# Launch comprehensive evaluation on AWS EC2
# Runs all models on all Nguyen benchmarks with PPO and GRPO
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
INSTANCE_TYPE="g5.2xlarge"  # Larger instance for parallel execution
AMI_ID="ami-0e86e20dae9224db8"  # Ubuntu 24.04 in us-east-1
KEY_NAME="chave-gpu-nova"
SECURITY_GROUP="sg-0deaa73e23482e3f6"
INSTANCE_NAME="seriguela-comprehensive-eval"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --wandb-key)
            WANDB_KEY="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults
EPOCHS=${EPOCHS:-20}
ALGORITHMS=${ALGORITHMS:-"ppo grpo"}

# Load credentials if not provided
if [ -z "$WANDB_KEY" ] || [ -z "$HF_TOKEN" ]; then
    if [ -f ~/.tokens.txt ]; then
        if [ -z "$HF_TOKEN" ]; then
            HF_TOKEN=$(grep "huggingface" ~/.tokens.txt | cut -d'=' -f2 | tr -d ' ')
        fi
        if [ -z "$WANDB_KEY" ]; then
            WANDB_KEY=$(grep "wandb" ~/.tokens.txt | cut -d'=' -f2 | tr -d ' ')
        fi
    fi
fi

# Validate credentials
if [ -z "$WANDB_KEY" ] || [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}Error: Missing credentials. Provide --wandb-key and --hf-token${NC}"
    exit 1
fi

echo -e "${GREEN}Launching comprehensive evaluation on AWS${NC}"
echo "Instance type: $INSTANCE_TYPE"
echo "Models: ${MODELS:-all}"
echo "Benchmarks: ${BENCHMARKS:-all}"
echo "Algorithms: $ALGORITHMS"
echo "Epochs: $EPOCHS"

# Create user data script (Windows-compatible path)
TEMP_DIR="${TMPDIR:-/tmp}"
if [ -d "/c/Users/madeinweb/temp" ]; then
    TEMP_DIR="/c/Users/madeinweb/temp"
fi
mkdir -p "$TEMP_DIR"

cat > "$TEMP_DIR/userdata_eval.sh" << 'EOF'
#!/bin/bash
exec > >(tee -a /home/ubuntu/setup.log)
exec 2>&1

echo "Starting setup at $(date)"

# Wait for cloud-init to complete (with timeout)
timeout 300 cloud-init status --wait || echo "cloud-init wait timed out"

# Update system
apt-get update
apt-get install -y python3-pip python3-venv git htop nvtop

# Install NVIDIA drivers if not present
if ! nvidia-smi; then
    apt-get install -y nvidia-driver-535
fi

# Switch to ubuntu user for the rest
su - ubuntu << 'EOFU'
cd ~

# Create virtual environment
python3 -m venv seriguela_env
source seriguela_env/bin/activate

# Clone repository
if [ ! -d "seriguela" ]; then
    git clone https://github.com/Agentes-I-A/Seriguela.git seriguela
fi

cd seriguela
git pull origin main

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib seaborn

# Set up credentials
EOF

# Add credentials to user data
cat >> "$TEMP_DIR/userdata_eval.sh" << EOF
export HUGGINGFACE_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$WANDB_KEY"

# Login to HuggingFace
huggingface-cli login --token \$HUGGINGFACE_TOKEN

# Login to Wandb
wandb login \$WANDB_API_KEY

# Create tokens file for scripts
echo "huggingface = \$HUGGINGFACE_TOKEN" > ~/.tokens.txt
echo "wandb = \$WANDB_API_KEY" >> ~/.tokens.txt

# Pull models from HuggingFace if needed
echo "Downloading models..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download infix model
print('Downloading infix model...')
model = AutoModelForCausalLM.from_pretrained('augustocsc/Se124M_700K_infix_v3_json',
                                            torch_dtype=torch.float16,
                                            trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('augustocsc/Se124M_700K_infix_v3_json')
print('Infix model downloaded')
"

# Run evaluation
echo "Starting comprehensive evaluation at \$(date)"

# Build command
CMD="python scripts/run_comprehensive_evaluation.py --output_dir ./evaluation_results --epochs $EPOCHS"

# Add optional parameters
EOF

# Add model/benchmark selection if specified
if [ -n "$MODELS" ]; then
    cat >> "$TEMP_DIR/userdata_eval.sh" << EOF
CMD="\$CMD --models $MODELS"
EOF
fi

if [ -n "$BENCHMARKS" ]; then
    cat >> "$TEMP_DIR/userdata_eval.sh" << EOF
CMD="\$CMD --benchmarks $BENCHMARKS"
EOF
fi

if [ "$QUICK_TEST" == "true" ]; then
    cat >> "$TEMP_DIR/userdata_eval.sh" << EOF
CMD="\$CMD --quick_test"
EOF
fi

cat >> "$TEMP_DIR/userdata_eval.sh" << EOF
CMD="\$CMD --algorithms $ALGORITHMS"

echo "Running: \$CMD"
nohup \$CMD > evaluation.log 2>&1 &

echo "Evaluation started in background. Check evaluation.log for progress."

# Also run analysis periodically
(
    while true; do
        sleep 300  # Every 5 minutes
        if [ -d "./evaluation_results" ]; then
            python scripts/analyze_evaluation_results.py --results_dir ./evaluation_results > analysis.log 2>&1
        fi
    done
) &

EOFU

# Mark completion
touch /home/ubuntu/.setup_complete
echo "Setup complete at \$(date)"
EOF

# Launch instance
echo -e "${YELLOW}Launching EC2 instance...${NC}"

# Convert path to Windows format if needed
USERDATA_PATH="$TEMP_DIR/userdata_eval.sh"
if [[ "$USERDATA_PATH" == /c/* ]]; then
    USERDATA_PATH=$(echo "$USERDATA_PATH" | sed 's|^/c/|C:/|')
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --user-data "file://$USERDATA_PATH" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}Instance launched: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}Instance is running!${NC}"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "SSH command:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Monitor setup:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'tail -f setup.log'"
echo ""
echo "Monitor evaluation:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'tail -f seriguela/evaluation.log'"
echo ""
echo "Check GPU:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'nvidia-smi'"
echo ""
echo "Download results when complete:"
echo "  scp -r -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP:~/seriguela/evaluation_results ./"
echo ""
echo -e "${YELLOW}IMPORTANT: Remember to stop the instance when done!${NC}"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"