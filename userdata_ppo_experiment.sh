#!/bin/bash
# =============================================================================
# AWS EC2 Userdata Script: PPO Symbolic Regression Experiment
# =============================================================================
# This script:
# 1. Sets up the environment (Python, CUDA, dependencies)
# 2. Clones the seriguela repo with the experiment branch
# 3. Prepares training data (JSON format)
# 4. Trains the base model (exp_a_json) - required for PPO
# 5. Runs PPO experiments on test datasets
# 6. Saves and uploads results
#
# Instance: g5.xlarge (NVIDIA A10G, 24GB VRAM)
# Estimated time: ~4-5 hours
# =============================================================================

exec > /var/log/user-data.log 2>&1
set -x

echo "=========================================="
echo "PPO SYMBOLIC REGRESSION EXPERIMENT"
echo "Started: $(date)"
echo "=========================================="

cloud-init status --wait

# Run everything as ubuntu user
sudo -u ubuntu bash << 'UBUNTUSETUP'
cd /home/ubuntu

LOG_FILE="/home/ubuntu/ppo_experiment.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# === Configuration ===
# IMPORTANT: Replace these placeholders before launching
# You can find your tokens at:
#   HuggingFace: https://huggingface.co/settings/tokens
#   W&B: https://wandb.ai/authorize
HF_TOKEN="__HF_TOKEN__"
WANDB_KEY="__WANDB_KEY__"
REPO_URL="https://github.com/augustocsc/seriguela.git"
BRANCH="experiment/ppo-symbolic-regression"
WORKDIR="/home/ubuntu/seriguela"

log "=========================================="
log "PPO SYMBOLIC REGRESSION EXPERIMENT"
log "=========================================="

# === System Setup ===
log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git

# === Clone Repository ===
log "Cloning repository..."
cd /home/ubuntu

if [ -d "$WORKDIR" ]; then
    log "Directory exists, pulling latest changes..."
    cd "$WORKDIR"
    git fetch origin
    git checkout "$BRANCH" || git checkout -b "$BRANCH" origin/"$BRANCH"
    git pull origin "$BRANCH"
else
    git clone "$REPO_URL" seriguela
    cd "$WORKDIR"
    git checkout "$BRANCH" || git checkout -b "$BRANCH" origin/"$BRANCH"
fi

# === Python Environment ===
log "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
log "Installing PyTorch with CUDA 12.1..."
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
log "Installing dependencies..."
pip install -r requirements.txt

# Install TRL for PPO
pip install trl>=0.7.0

# Verify GPU
log "Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# === HuggingFace and W&B Login ===
if [ "$HF_TOKEN" != "YOUR_HF_TOKEN" ]; then
    log "Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
fi

if [ "$WANDB_KEY" != "YOUR_WANDB_KEY" ]; then
    log "Logging into Weights & Biases..."
    wandb login "$WANDB_KEY"
fi

# === Prepare Data ===
log "=========================================="
log "PHASE 1: Preparing Training Data"
log "=========================================="

python scripts/data/prepare_experiment_data.py \
    --dataset_repo_id augustocsc/sintetico_natural \
    --data_dir 700K \
    --data_column i_prompt_n \
    --output_base_dir ./data/experiments

# Create PPO test datasets
log "Creating PPO test datasets..."
python scripts/data/create_ppo_test_datasets.py

# === Train Base Model ===
log "=========================================="
log "PHASE 2: Training Base Model (exp_a_json)"
log "=========================================="

# Check if model already exists
if [ -d "./output/exp_a_json" ] && [ -f "./output/exp_a_json/adapter_config.json" ]; then
    log "Base model already exists, skipping training..."
else
    log "Training base model with JSON format..."

    python scripts/train_experiment.py \
        --experiment_name exp_a_json \
        --train_file ./data/experiments/exp_a_json/train.csv \
        --validation_file ./data/experiments/exp_a_json/validation.csv \
        --output_dir ./output/exp_a_json \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --learning_rate 5e-5 \
        --warmup_steps 500 \
        --save_steps 5000 \
        --logging_steps 100 \
        --wandb_project seriguela_ppo_exp 2>&1 | tee /home/ubuntu/base_model_training.log

    log "Base model training complete."
fi

# === Evaluate Base Model ===
log "=========================================="
log "PHASE 3: Evaluating Base Model"
log "=========================================="

python scripts/evaluate_experiments.py \
    --model_path ./output/exp_a_json \
    --num_samples 100 \
    --output_file ./output/exp_a_json/evaluation_results.json 2>&1 | tee /home/ubuntu/base_model_eval.log

# === Run PPO Experiments ===
log "=========================================="
log "PHASE 4: Running PPO Experiments"
log "=========================================="

log "Starting PPO experiments on test datasets..."

python scripts/run_ppo_experiments.py \
    --model_path ./output/exp_a_json \
    --batch_size 32 \
    --epochs 10 \
    --baseline_samples 200 2>&1 | tee /home/ubuntu/ppo_experiments.log

log "PPO experiments complete."

# === Single Dataset Deep Experiment ===
log "=========================================="
log "PHASE 5: Deep PPO Experiment (mul_x1_x2)"
log "=========================================="

# Run longer experiment on the mul_x1_x2 dataset
python scripts/ppo_experiment.py \
    --model_path ./output/exp_a_json \
    --dataset ./data/ppo_test/mul_x1_x2.csv \
    --output_dir ./output/ppo_deep_mul \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-5 \
    --early_stop_r2 0.99 2>&1 | tee /home/ubuntu/ppo_deep.log

# === Compile Results ===
log "=========================================="
log "PHASE 6: Compiling Results"
log "=========================================="

# Create results summary
cat > /home/ubuntu/EXPERIMENT_SUMMARY.md << 'EOF'
# PPO Symbolic Regression Experiment Summary

## Experiment Date
$(date '+%Y-%m-%d %H:%M:%S')

## Configuration
- Instance: g5.xlarge (NVIDIA A10G)
- Base Model: exp_a_json (JSON format, 80% valid)
- PPO Epochs: 10 (quick), 20 (deep)
- Batch Size: 32-64

## Results Location
- Base model: ./output/exp_a_json/
- PPO experiments: ./output/ppo_experiments/
- Deep experiment: ./output/ppo_deep_mul/
- Logs: /home/ubuntu/*.log

## Key Questions
1. Does PPO improve R² scores over baseline?
2. Can PPO find the exact expression?
3. How many epochs are needed?

## Files to Download
- ./output/ppo_experiments/*/summary.json
- ./output/ppo_deep_mul/final_results_*.json
- /home/ubuntu/ppo_experiments.log
EOF

log "Results summary created."

# === Mark Complete ===
touch /home/ubuntu/.ppo_experiment_complete
log "=========================================="
log "EXPERIMENT COMPLETE"
log "=========================================="
log "Check /home/ubuntu/EXPERIMENT_SUMMARY.md for results"
log "Download results with:"
log "  scp -i key.pem ubuntu@IP:/home/ubuntu/seriguela/output/ppo_experiments/*/summary.json ."

# Keep instance running for result retrieval
log "Instance will remain running. Stop manually after downloading results."

UBUNTUSETUP

echo "Userdata script finished at $(date)"
