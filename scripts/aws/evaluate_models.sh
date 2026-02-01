#!/bin/bash
# Script to evaluate two models on AWS and compare results
# This script compares the original model (without end token) with the v2 model (with end token)
# Usage: bash scripts/aws/evaluate_models.sh

set -e

echo "=========================================="
echo "Model Comparison: v1 vs v2"
echo "=========================================="
echo "Model 1: augustocsc/Se124M_700K_infix (original)"
echo "Model 2: augustocsc/Se124M_700K_infix_v2 (with <|endofex|> token)"
echo "=========================================="
echo ""

# Activate virtual environment
source ~/seriguela/venv/bin/activate
cd ~/seriguela

# Set up logging
LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date)] Starting evaluation..."
echo ""

# Check GPU availability
echo "Checking GPU..."
if nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: No GPU detected. Evaluation will be slow."
    echo ""
fi

# Run comparison
echo "Running model comparison..."
echo "This will evaluate both models on 500 samples from the test set."
echo ""

python scripts/compare_models.py \
    --model1 augustocsc/Se124M_700K_infix \
    --model2 augustocsc/Se124M_700K_infix_v2 \
    --model1_name "Original (no end token)" \
    --model2_name "V2 (with <|endofex|>)" \
    --num_samples 500 \
    --dataset_repo_id augustocsc/sintetico_natural \
    --data_dir 700K \
    --data_column i_prompt_n \
    --output_dir ./evaluation_results/comparison

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: ./evaluation_results/comparison"
echo "Log file: $LOG_FILE"
echo ""
echo "To view results:"
echo "  cat ./evaluation_results/comparison/comparison_*.json | jq"
echo ""
