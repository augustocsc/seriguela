#!/bin/bash
# EXP-A: Training with JSON structured format
# Uses <|endofex|> as end marker

set -e

echo "=============================================="
echo "EXP-A: JSON Format Training"
echo "=============================================="

cd ~/seriguela

# Activate virtual environment
source venv/bin/activate

# Check data exists
if [ ! -f "./data/experiments/exp_a_json/train.csv" ]; then
    echo "ERROR: Training data not found!"
    echo "Expected: ./data/experiments/exp_a_json/train.csv"
    exit 1
fi

# Count samples
TRAIN_COUNT=$(wc -l < ./data/experiments/exp_a_json/train.csv)
echo "Training samples: $TRAIN_COUNT"

# Training configuration
export WANDB_PROJECT="seriguela_experiments"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Run training
echo ""
echo "Starting training..."
echo "Output: ./output/exp_a_json"
echo ""

python scripts/train_experiment.py \
    --experiment_name "exp_a_json" \
    --train_file ./data/experiments/exp_a_json/train.csv \
    --validation_file ./data/experiments/exp_a_json/validation.csv \
    --output_dir ./output/exp_a_json \
    --end_marker "<|endofex|>" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --block_size 256 \
    --fp16 \
    --wandb_project seriguela_experiments \
    --wandb_run_name "exp_a_json_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=============================================="
echo "EXP-A Training Complete!"
echo "=============================================="
echo "Model saved to: ./output/exp_a_json"
