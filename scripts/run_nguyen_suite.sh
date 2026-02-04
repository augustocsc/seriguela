#!/bin/bash
# Run complete Nguyen benchmark suite (1-12) on all models with all algorithms
# 3 models × 12 benchmarks × 4 algorithms = 144 experiments
# Part of Model Scaling Experiment (Feb 2025)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "Nguyen Benchmark Suite"
echo "Model Scaling Experiment"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Models:     3 (Base, Medium, Large)"
echo "  Benchmarks: 12 (Nguyen 1-12)"
echo "  Algorithms: 4 (Supervised, REINFORCE, GRPO, PPO)"
echo "  Total:      144 experiments"
echo ""

# Configuration
MODELS=("base" "medium" "large")
BENCHMARKS=(1 2 3 4 5 6 7 8 9 10 11 12)
ALGORITHMS=("supervised" "reinforce" "grpo" "ppo")

OUTPUT_BASE="./nguyen_suite_results"
DATA_DIR="./data/benchmarks/nguyen"

# Algorithm parameters
RL_EPOCHS=20
RL_SAMPLES_PER_EPOCH=100
SUPERVISED_SAMPLES=200

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Counters
TOTAL_EXPERIMENTS=144
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Create experiment log
LOG_FILE="$OUTPUT_BASE/experiment_log.txt"
echo "Nguyen Suite Experiment Log" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Total experiments: $TOTAL_EXPERIMENTS" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

print_status "Starting Nguyen suite evaluation..."
echo ""

# Function to run experiment
run_experiment() {
    local model=$1
    local bench=$2
    local algo=$3
    local exp_num=$4

    local model_path="./output/gpt2_${model}_700K_json"
    local dataset="$DATA_DIR/nguyen_${bench}.csv"
    local output_file="$OUTPUT_BASE/${model}_nguyen${bench}_${algo}.json"

    # Check if dataset exists
    if [ ! -f "$dataset" ]; then
        print_warning "Dataset not found: $dataset (skipping)"
        echo "[$exp_num/$TOTAL_EXPERIMENTS] SKIPPED: $model + Nguyen-$bench + $algo (dataset missing)" >> "$LOG_FILE"
        return 1
    fi

    # Check if model exists
    if [ ! -d "$model_path" ]; then
        print_error "Model not found: $model_path (skipping)"
        echo "[$exp_num/$TOTAL_EXPERIMENTS] FAILED: $model + Nguyen-$bench + $algo (model missing)" >> "$LOG_FILE"
        return 1
    fi

    echo -ne "${BLUE}[$exp_num/$TOTAL_EXPERIMENTS]${NC} Running: ${model} + Nguyen-${bench} + ${algo}..."

    local start=$(date +%s)
    local success=0

    case $algo in
        supervised)
            python scripts/evaluate.py \
                --model_path "$model_path" \
                --dataset "$dataset" \
                --num_samples $SUPERVISED_SAMPLES \
                --output_file "$output_file" \
                > "$OUTPUT_BASE/${model}_nguyen${bench}_${algo}.log" 2>&1 && success=1
            ;;
        reinforce)
            python scripts/reinforce_symbolic.py \
                --model_path "$model_path" \
                --dataset "$dataset" \
                --epochs $RL_EPOCHS \
                --samples_per_epoch $RL_SAMPLES_PER_EPOCH \
                --output_file "$output_file" \
                > "$OUTPUT_BASE/${model}_nguyen${bench}_${algo}.log" 2>&1 && success=1
            ;;
        grpo)
            python scripts/grpo_symbolic.py \
                --model_path "$model_path" \
                --dataset "$dataset" \
                --epochs $RL_EPOCHS \
                --samples_per_epoch $RL_SAMPLES_PER_EPOCH \
                --output_file "$output_file" \
                > "$OUTPUT_BASE/${model}_nguyen${bench}_${algo}.log" 2>&1 && success=1
            ;;
        ppo)
            python scripts/ppo_symbolic.py \
                --model_path "$model_path" \
                --dataset "$dataset" \
                --epochs $RL_EPOCHS \
                --samples_per_epoch $RL_SAMPLES_PER_EPOCH \
                --output_file "$output_file" \
                > "$OUTPUT_BASE/${model}_nguyen${bench}_${algo}.log" 2>&1 && success=1
            ;;
    esac

    local end=$(date +%s)
    local duration=$((end - start))

    if [ $success -eq 1 ]; then
        echo -e " ${GREEN}✓${NC} (${duration}s)"
        echo "[$exp_num/$TOTAL_EXPERIMENTS] SUCCESS: $model + Nguyen-$bench + $algo (${duration}s)" >> "$LOG_FILE"
        return 0
    else
        echo -e " ${RED}✗${NC} (${duration}s)"
        echo "[$exp_num/$TOTAL_EXPERIMENTS] FAILED: $model + Nguyen-$bench + $algo (${duration}s)" >> "$LOG_FILE"
        return 1
    fi
}

# Run all experiments
exp_num=0

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Model: GPT-2 ${model^}"
    echo "=========================================="

    for bench in "${BENCHMARKS[@]}"; do
        for algo in "${ALGORITHMS[@]}"; do
            ((exp_num++))

            if run_experiment "$model" "$bench" "$algo" "$exp_num"; then
                ((COMPLETED++))
            else
                ((FAILED++))
            fi

            # Progress summary every 12 experiments
            if [ $((exp_num % 12)) -eq 0 ]; then
                echo ""
                print_status "Progress: $exp_num/$TOTAL_EXPERIMENTS ($(((exp_num * 100) / TOTAL_EXPERIMENTS))%)"
                print_status "Completed: $COMPLETED | Failed: $FAILED"
                echo ""
            fi
        done
    done
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "Nguyen Suite Completed!"
echo "=========================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Completed:         $COMPLETED"
echo "Failed:            $FAILED"
echo "Success rate:      $(((COMPLETED * 100) / TOTAL_EXPERIMENTS))%"
echo "Total time:        ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to:  $OUTPUT_BASE/"
echo "Log file:          $LOG_FILE"
echo ""

# Update log file
echo "" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"
echo "Finished: $(date)" >> "$LOG_FILE"
echo "Total duration: ${HOURS}h ${MINUTES}m" >> "$LOG_FILE"
echo "Completed: $COMPLETED" >> "$LOG_FILE"
echo "Failed: $FAILED" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"

print_status "Next step: Aggregate results"
echo ""
echo "Run:"
echo "  python scripts/aggregate_nguyen_results.py --input_dir $OUTPUT_BASE"
echo ""

if [ $FAILED -gt 0 ]; then
    print_warning "Some experiments failed. Check logs in $OUTPUT_BASE/"
    exit 1
fi

exit 0
