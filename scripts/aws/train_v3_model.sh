#!/bin/bash
# Training script for v3 model with proper end markers
# This script is designed to be run on AWS EC2 instances with GPU

set -e  # Exit on error

echo "=================================================="
echo "Seriguela v3 Model Training"
echo "=================================================="
echo "Start time: $(date)"
echo ""

# Configuration
PROJECT_DIR="${HOME}/seriguela"
OUTPUT_DIR="${PROJECT_DIR}/output/Se124M_700K_infix_v3"
CONFIG_FILE="${PROJECT_DIR}/configs/training_v3.json"
DATA_DIR="${PROJECT_DIR}/data/processed/700K_fixed"

# Check if running in project directory
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".seriguela" ]; then
    source .seriguela/bin/activate
else
    echo "ERROR: Virtual environment not found!"
    exit 1
fi

# Verify GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "WARNING: GPU not detected! Training will be slow on CPU."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify data files exist
echo ""
echo "Verifying training data..."
if [ ! -f "$DATA_DIR/train_700K.csv" ]; then
    echo "ERROR: Training data not found: $DATA_DIR/train_700K.csv"
    echo "Please ensure data preparation step was completed."
    exit 1
fi

if [ ! -f "$DATA_DIR/validation_700K.csv" ]; then
    echo "ERROR: Validation data not found: $DATA_DIR/validation_700K.csv"
    exit 1
fi

# Check for end markers in data
echo "Checking for end markers in training data..."
MARKER_COUNT=$(head -100 "$DATA_DIR/train_700K.csv" | grep -c "<|endofex|>" || true)
if [ "$MARKER_COUNT" -eq 0 ]; then
    echo "ERROR: No <|endofex|> markers found in training data!"
    echo "Please run data preparation script first."
    exit 1
else
    echo "✓ End markers detected in training data"
fi

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Training data: $DATA_DIR/train_700K.csv"
echo "  Validation data: $DATA_DIR/validation_700K.csv"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export WANDB_PROJECT="seriguela_v3"
export WANDB_RUN_NAME="v3_proper_markers_$(date +%Y%m%d_%H%M%S)"

# Check if wandb is configured
if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
    echo "WARNING: Weights & Biases not configured. Training will proceed without W&B logging."
    echo "To enable W&B: wandb login"
fi

# Start training
echo ""
echo "=================================================="
echo "Starting training..."
echo "=================================================="
echo ""

# Run training with config file
python scripts/train.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --use_local_csvs \
    --train_file "$DATA_DIR/train_700K.csv" \
    --validation_file "$DATA_DIR/validation_700K.csv" \
    --wandb_project seriguela_v3 \
    --wandb_run_name "$WANDB_RUN_NAME"

TRAIN_EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training completed"
echo "=================================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Run evaluation: python scripts/evaluate.py --model_path $OUTPUT_DIR"
    echo "2. Test generation: python scripts/generate.py --model_path $OUTPUT_DIR --num_generations 50 --validate"
    echo "3. Push to Hub (if configured): huggingface-cli upload augustocsc/Se124M_700K_infix_v3 $OUTPUT_DIR"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check logs above for error details."
    exit $TRAIN_EXIT_CODE
fi
