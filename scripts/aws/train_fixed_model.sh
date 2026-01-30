#!/bin/bash
# Train model with proper end-of-expression markers
# This script retrains the Seriguela model with <|endofex|> markers in the training data
# so the model learns to stop generation correctly.

set -e  # Exit on error

echo "================================================================"
echo "SERIGUELA - Training Model with Proper End Markers"
echo "================================================================"

# Configuration
MODEL_NAME="gpt2"
DATASET_REPO="augustocsc/sintetico_natural"
DATA_DIR="700K"
DATA_COLUMN="i_prompt_n"  # or p_prompt_n for prefix
OUTPUT_DIR="./output/Se124M_700K_infix_v2"
HUB_MODEL_ID="augustocsc/Se124M_700K_infix_v2"  # NEW REPO NAME

# Hyperparameters
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=5e-5
BLOCK_SIZE=128
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.05

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_REPO/$DATA_DIR"
echo "  Data Column: $DATA_COLUMN"
echo "  Output: $OUTPUT_DIR"
echo "  Hub Model: $HUB_MODEL_ID"
echo ""
echo "Hyperparameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Block Size: $BLOCK_SIZE"
echo "  LoRA r: $LORA_R"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  LoRA dropout: $LORA_DROPOUT"
echo "================================================================"

# Check if data preparation is needed
echo ""
echo "[Step 1/3] Checking data preparation..."
if [ ! -f "./data/processed/700K_fixed/train_700K.csv" ]; then
    echo "Training data not found. Preparing data with end markers..."

    python scripts/data/prepare_training_data_fixed.py \
        --dataset_repo_id $DATASET_REPO \
        --data_dir $DATA_DIR \
        --data_column $DATA_COLUMN \
        --output_dir ./data/processed/700K_fixed \
        --validate

    if [ $? -ne 0 ]; then
        echo "❌ Data preparation failed!"
        exit 1
    fi

    echo "✅ Data preparation complete!"
else
    echo "✅ Training data already prepared (./data/processed/700K_fixed/)"
fi

# Optional: Show sample of prepared data
echo ""
echo "Sample of prepared data:"
head -n 2 ./data/processed/700K_fixed/train_700K.csv
echo ""

# Start training
echo ""
echo "[Step 2/3] Starting training..."
echo "================================================================"
echo ""

python scripts/train.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_repo_id $DATASET_REPO \
    --data_dir $DATA_DIR \
    --data_column $DATA_COLUMN \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --block_size $BLOCK_SIZE \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --push_to_hub \
    --hub_model_id $HUB_MODEL_ID \
    --logging_steps 100 \
    --seed 42

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Training complete!"

# Quick test generation
echo ""
echo "[Step 3/3] Testing model generation..."
echo "================================================================"
echo ""

python scripts/generate.py \
    --model_path $OUTPUT_DIR \
    --num_generations 5 \
    --validate

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ Generation test failed, but model was trained successfully"
else
    echo ""
    echo "✅ Generation test passed!"
fi

# Summary
echo ""
echo "================================================================"
echo "TRAINING COMPLETE"
echo "================================================================"
echo "Model saved to: $OUTPUT_DIR"
echo "Model pushed to: $HUB_MODEL_ID"
echo ""
echo "Next steps:"
echo "  1. Evaluate the model: python scripts/evaluate.py --model_path $OUTPUT_DIR"
echo "  2. Compare with old model: python scripts/compare_models.py --model1 ./output/Se124M_700K_infix --model2 $OUTPUT_DIR"
echo "  3. Generate more samples: python scripts/generate.py --model_path $OUTPUT_DIR --num_generations 20"
echo "================================================================"
