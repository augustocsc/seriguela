#!/bin/bash
# Workflow completo de treinamento para AWS g5.xlarge
# Projeto Seriguela - Treinar 6 modelos GPT-2 (3 tamanhos x 2 formatos)

set -e  # Exit on error

echo "=========================================="
echo "Seriguela - Full Training Workflow"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "$1"
    echo -e "==========================================${NC}"
    echo ""
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment not activated. Activating..."
    source venv/bin/activate 2>/dev/null || {
        print_error "Could not activate virtual environment. Please run setup_aws.sh first."
        exit 1
    }
fi

# Check environment variables
if [ -z "$HF_TOKEN" ]; then
    print_error "HF_TOKEN not set. Please export HF_TOKEN='your_token'"
    exit 1
fi

# Check GPU
print_status "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || {
    print_error "GPU not available!"
    exit 1
}

# Dataset configuration
DATASET_REPO="augustocsc/sintetico_natural"
DATA_DIR="700K"
HF_USER="augustocsc"

# Common training parameters
WANDB_PROJECT="seriguela_700K"
SEED=42
BLOCK_SIZE=128

# Output directories
OUTPUT_BASE="./output"
EVAL_OUTPUT="./evaluation_results"
mkdir -p "$OUTPUT_BASE" "$EVAL_OUTPUT"

# Training configurations
# Format: "model_name|epochs|batch_size|grad_accum|learning_rate|run_suffix"
declare -a CONFIGS=(
    # GPT-2 Small (124M)
    "gpt2|3|16|4|5e-5|Se124M"
    # GPT-2 Medium (355M)
    "gpt2-medium|2|8|8|3e-5|Se355M"
    # GPT-2 Large (774M)
    "gpt2-large|2|4|16|2e-5|Se774M"
)

# Data columns for formats
declare -a DATA_COLUMNS=(
    "i_prompt_n|infix"
    "p_prompt_n|prefix"
)

# Function to run training
run_training() {
    local model_name=$1
    local epochs=$2
    local batch_size=$3
    local grad_accum=$4
    local lr=$5
    local run_suffix=$6
    local data_column=$7
    local format=$8

    local run_name="${run_suffix}_${DATA_DIR}_${format}"
    local output_dir="${OUTPUT_BASE}/${run_name}"
    local hub_model_id="${HF_USER}/${run_name}"

    print_header "Training: $run_name"
    echo "Model: $model_name"
    echo "Epochs: $epochs"
    echo "Batch size: $batch_size"
    echo "Gradient accumulation: $grad_accum"
    echo "Effective batch size: $((batch_size * grad_accum))"
    echo "Learning rate: $lr"
    echo "Data column: $data_column"
    echo "Output: $output_dir"
    echo "Hub ID: $hub_model_id"
    echo ""

    # Run training
    python scripts/train.py \
        --model_name_or_path "$model_name" \
        --dataset_repo_id "$DATASET_REPO" \
        --data_dir "$DATA_DIR" \
        --data_column "$data_column" \
        --approach "$format" \
        --output_dir "$output_dir" \
        --num_train_epochs "$epochs" \
        --per_device_train_batch_size "$batch_size" \
        --per_device_eval_batch_size "$batch_size" \
        --gradient_accumulation_steps "$grad_accum" \
        --learning_rate "$lr" \
        --weight_decay 0.01 \
        --warmup_steps 100 \
        --block_size "$BLOCK_SIZE" \
        --logging_steps 50 \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --fp16 \
        --seed "$SEED" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$run_name" \
        --push_to_hub \
        --hub_model_id "$hub_model_id"

    # Check if training was successful
    if [ $? -eq 0 ]; then
        print_status "Training completed successfully: $run_name"
        return 0
    else
        print_error "Training failed: $run_name"
        return 1
    fi
}

# Function to run evaluation
run_evaluation() {
    local model_path=$1
    local data_column=$2
    local num_samples=${3:-500}

    print_status "Evaluating model: $model_path"

    python scripts/evaluate.py \
        --model_path "$model_path" \
        --dataset_repo_id "$DATASET_REPO" \
        --data_dir "$DATA_DIR" \
        --data_column "$data_column" \
        --num_samples "$num_samples" \
        --output_dir "$EVAL_OUTPUT" \
        --temperature 0.7 \
        --seed "$SEED"

    if [ $? -eq 0 ]; then
        print_status "Evaluation completed: $model_path"
    else
        print_warning "Evaluation had issues: $model_path"
    fi
}

# Parse command line arguments
RUN_TEST=false
RUN_TRAINING=true
RUN_EVAL=true
SPECIFIC_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            RUN_TEST=true
            RUN_TRAINING=false
            RUN_EVAL=false
            shift
            ;;
        --no-eval)
            RUN_EVAL=false
            shift
            ;;
        --eval-only)
            RUN_TRAINING=false
            RUN_EVAL=true
            shift
            ;;
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test-only    Run only the test training (1 epoch)"
            echo "  --no-eval      Skip evaluation after training"
            echo "  --eval-only    Run only evaluation (skip training)"
            echo "  --model NAME   Train only specific model (gpt2, gpt2-medium, gpt2-large)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Test run
if [ "$RUN_TEST" = true ]; then
    print_header "Running Test Training (1 epoch with gpt2)"

    python scripts/train.py \
        --model_name_or_path gpt2 \
        --dataset_repo_id "$DATASET_REPO" \
        --data_dir "$DATA_DIR" \
        --data_column "i_prompt_n" \
        --approach "infix" \
        --output_dir "${OUTPUT_BASE}/test_run" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --block_size "$BLOCK_SIZE" \
        --logging_steps 20 \
        --eval_strategy epoch \
        --save_strategy epoch \
        --fp16 \
        --seed "$SEED" \
        --wandb_project "${WANDB_PROJECT}_test"

    print_status "Test training completed!"
    print_status "Checklist:"
    echo "  [ ] GPU detected and functioning"
    echo "  [ ] Dataset loaded correctly"
    echo "  [ ] Training completed without errors"
    echo "  [ ] Wandb received metrics"
    echo "  [ ] Model saved locally"
    echo ""
    echo "Now test evaluate.py and generate.py:"
    echo "  python scripts/evaluate.py --model_path ./output/test_run --num_samples 50"
    echo "  python scripts/generate.py --model_path ./output/test_run --num_generations 5 --validate"
    exit 0
fi

# Track completed trainings
declare -a COMPLETED_MODELS=()
declare -a FAILED_MODELS=()

# Main training loop
if [ "$RUN_TRAINING" = true ]; then
    print_header "Starting Full Training Workflow"

    START_TIME=$(date +%s)

    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r model_name epochs batch_size grad_accum lr run_suffix <<< "$config"

        # Skip if specific model requested and this is not it
        if [ -n "$SPECIFIC_MODEL" ] && [ "$model_name" != "$SPECIFIC_MODEL" ]; then
            continue
        fi

        for data_config in "${DATA_COLUMNS[@]}"; do
            IFS='|' read -r data_column format <<< "$data_config"

            run_name="${run_suffix}_${DATA_DIR}_${format}"

            print_status "Starting training: $run_name"

            if run_training "$model_name" "$epochs" "$batch_size" "$grad_accum" "$lr" "$run_suffix" "$data_column" "$format"; then
                COMPLETED_MODELS+=("${HF_USER}/${run_name}|${data_column}")
            else
                FAILED_MODELS+=("$run_name")
            fi

            # Small delay between trainings
            sleep 10
        done
    done

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    print_header "Training Summary"
    echo "Total time: ${HOURS}h ${MINUTES}m"
    echo ""
    echo "Completed models (${#COMPLETED_MODELS[@]}):"
    for model in "${COMPLETED_MODELS[@]}"; do
        echo "  - ${model%|*}"
    done
    echo ""
    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        echo "Failed models (${#FAILED_MODELS[@]}):"
        for model in "${FAILED_MODELS[@]}"; do
            echo "  - $model"
        done
    fi
fi

# Evaluation
if [ "$RUN_EVAL" = true ]; then
    print_header "Running Evaluations"

    # If we just trained, use those models
    if [ ${#COMPLETED_MODELS[@]} -gt 0 ]; then
        for model_info in "${COMPLETED_MODELS[@]}"; do
            IFS='|' read -r model_path data_column <<< "$model_info"
            run_evaluation "$model_path" "$data_column" 500
        done
    else
        # Otherwise, evaluate all expected models
        for config in "${CONFIGS[@]}"; do
            IFS='|' read -r model_name epochs batch_size grad_accum lr run_suffix <<< "$config"

            for data_config in "${DATA_COLUMNS[@]}"; do
                IFS='|' read -r data_column format <<< "$data_config"

                run_name="${run_suffix}_${DATA_DIR}_${format}"
                model_path="${HF_USER}/${run_name}"

                run_evaluation "$model_path" "$data_column" 500
            done
        done
    fi

    print_header "Evaluation Complete"
    echo "Results saved to: $EVAL_OUTPUT"
fi

print_header "Workflow Complete!"
echo ""
echo "Next steps:"
echo "1. Check training results on wandb: https://wandb.ai/${WANDB_PROJECT}"
echo "2. Check models on HuggingFace Hub: https://huggingface.co/${HF_USER}"
echo "3. Review evaluation results in: $EVAL_OUTPUT"
echo ""
echo "To test a model interactively:"
echo "  python scripts/generate.py --model_path ${HF_USER}/Se124M_700K_infix --interactive --validate"
echo ""
