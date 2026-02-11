#!/bin/bash
# Fix instance setup by manually cloning repository and running evaluation

INSTANCE_IP="3.81.72.206"
SSH_KEY="C:/Users/madeinweb/chave-gpu.pem"

echo "Connecting to instance to fix setup..."

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'REMOTE_COMMANDS'

cd ~

# Clone repository (public URL)
if [ ! -d "seriguela" ]; then
    echo "Cloning Seriguela repository..."
    git clone https://github.com/Agentes-I-A/Seriguela.git seriguela
fi

cd seriguela

# Checkout correct branch
git fetch origin
git checkout experiment/ppo-symbolic-regression || git checkout -b experiment/ppo-symbolic-regression origin/experiment/ppo-symbolic-regression
git pull origin experiment/ppo-symbolic-regression

# Activate virtual environment
source ~/seriguela_env/bin/activate

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Install additional packages
pip install matplotlib seaborn

# Setup credentials from environment
if [ ! -f ~/.tokens.txt ]; then
    echo "huggingface = $HUGGINGFACE_TOKEN" > ~/.tokens.txt
    echo "wandb = $WANDB_API_KEY" >> ~/.tokens.txt
fi

# Login to services
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token $HUGGINGFACE_TOKEN
fi

if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
fi

# Download infix model from HuggingFace
echo "Downloading infix model..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Downloading infix model from HuggingFace...')
try:
    model = AutoModelForCausalLM.from_pretrained(
        'augustocsc/Se124M_700K_infix_v3_json',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('augustocsc/Se124M_700K_infix_v3_json')
    print('✓ Infix model downloaded successfully')
except Exception as e:
    print(f'✗ Error downloading model: {e}')
"

# Check GPU
echo "Checking GPU availability..."
nvidia-smi

# Create output directory
mkdir -p evaluation_results

# Start evaluation
echo "Starting comprehensive evaluation..."
nohup python scripts/run_comprehensive_evaluation.py \
    --output_dir ./evaluation_results \
    --epochs 20 \
    --algorithms ppo grpo \
    > evaluation.log 2>&1 &

echo "Evaluation started! PID: $!"
echo "Monitor with: tail -f ~/seriguela/evaluation.log"

# Also start periodic analysis
(
    while true; do
        sleep 300  # Every 5 minutes
        if [ -d "./evaluation_results" ]; then
            python scripts/analyze_evaluation_results.py --results_dir ./evaluation_results > analysis.log 2>&1 || true
        fi
    done
) &

echo "Setup fixed! Evaluation running in background."

REMOTE_COMMANDS

echo "Done! Instance is now running evaluation."
echo "Monitor with: ssh -i $SSH_KEY ubuntu@$INSTANCE_IP 'tail -f seriguela/evaluation.log'"
