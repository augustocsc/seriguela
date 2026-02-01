#!/bin/bash
# Complete setup and training script for EXP-B (EOS format)
# Run this on a fresh AWS instance

set -e

echo "=============================================="
echo "EXP-B: Complete Setup and Training"
echo "EOS Format with <|endoftext|> marker"
echo "=============================================="
echo "Started: $(date)"
echo ""

cd /home/ubuntu/seriguela

# Activate environment
source venv/bin/activate

# Step 1: Prepare data
echo "[1/3] Preparing training data..."
echo "This will download from HuggingFace Hub and convert to EOS format"
echo ""

mkdir -p data/experiments

python scripts/data/prepare_experiment_data.py \
    --dataset_repo_id augustocsc/sintetico_natural \
    --data_dir 700K \
    --data_column i_prompt_n \
    --output_base_dir ./data/experiments

# Verify data
if [ ! -f "./data/experiments/exp_b_eos/train.csv" ]; then
    echo "ERROR: Data preparation failed!"
    exit 1
fi

TRAIN_COUNT=$(wc -l < ./data/experiments/exp_b_eos/train.csv)
echo "Training samples: $TRAIN_COUNT"

# Step 2: Run training
echo ""
echo "[2/3] Starting training..."
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

# Step 3: Evaluate
echo ""
echo "[3/3] Evaluating model..."
echo ""

python scripts/evaluate_experiments.py \
    --model_path ./output/exp_b_eos \
    --experiment_type eos \
    --num_samples 200 \
    --output_file ./output/exp_b_eos/evaluation_results.json

echo ""
echo "=============================================="
echo "EXP-B Complete!"
echo "=============================================="
echo "Finished: $(date)"
echo "Model: ./output/exp_b_eos"
echo "Results: ./output/exp_b_eos/evaluation_results.json"

# Create completion marker
touch /home/ubuntu/.exp_b_complete
