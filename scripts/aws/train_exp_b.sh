#!/bin/bash
# EXP-B: Training with GPT-2 EOS token (<|endoftext|>)
# Uses native GPT-2 EOS token (ID 50256)

set -e

echo "=============================================="
echo "EXP-B: EOS Token Format Training"
echo "=============================================="

cd ~/seriguela

# Activate virtual environment
source venv/bin/activate

# Check data exists
if [ ! -f "./data/experiments/exp_b_eos/train.csv" ]; then
    echo "ERROR: Training data not found!"
    echo "Expected: ./data/experiments/exp_b_eos/train.csv"
    exit 1
fi

# Count samples
TRAIN_COUNT=$(wc -l < ./data/experiments/exp_b_eos/train.csv)
echo "Training samples: $TRAIN_COUNT"

# Training configuration
export WANDB_PROJECT="seriguela_experiments"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Run training
echo ""
echo "Starting training..."
echo "Output: ./output/exp_b_eos"
echo ""

python scripts/train_experiment.py \
    --experiment_name "exp_b_eos" \
    --train_file ./data/experiments/exp_b_eos/train.csv \
    --validation_file ./data/experiments/exp_b_eos/validation.csv \
    --output_dir ./output/exp_b_eos \
    --end_marker "<|endoftext|>" \
    --use_native_eos \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --block_size 128 \
    --fp16 \
    --wandb_project seriguela_experiments \
    --wandb_run_name "exp_b_eos_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=============================================="
echo "EXP-B Training Complete!"
echo "=============================================="
echo "Model saved to: ./output/exp_b_eos"
