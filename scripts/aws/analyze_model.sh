#!/bin/bash
# Automatic Model Analysis Script
# Runs evaluation and generation analysis after training

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_header() { echo -e "\n${BLUE}========================================\n$1\n========================================${NC}\n"; }

# Parameters
MODEL_PATH="${1:-./output/Se124M_700K_infix}"
DATA_COLUMN="${2:-i_prompt_n}"
DATASET_REPO="augustocsc/sintetico_natural"
DATA_DIR="700K"
NUM_SAMPLES=500
NUM_GENERATIONS=100

# Directories
PROJECT_DIR="/home/ubuntu/seriguela"
OUTPUT_DIR="$HOME/analysis_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"
source venv/bin/activate

print_header "Automatic Model Analysis"
print_status "Model: $MODEL_PATH"
print_status "Output: $OUTPUT_DIR"
echo ""

# =============================================================================
# 1. EVALUATE MODEL
# =============================================================================
print_header "Step 1: Model Evaluation"
print_status "Running evaluation on $NUM_SAMPLES samples..."

python scripts/evaluate.py \
    --model_path "$MODEL_PATH" \
    --dataset_repo_id "$DATASET_REPO" \
    --data_dir "$DATA_DIR" \
    --data_column "$DATA_COLUMN" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR/evaluation" \
    --temperature 0.7 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

if [ $? -eq 0 ]; then
    print_status "✅ Evaluation completed"
else
    print_status "⚠️ Evaluation had issues"
fi

# =============================================================================
# 2. GENERATE SAMPLES
# =============================================================================
print_header "Step 2: Sample Generation & Validation"
print_status "Generating $NUM_GENERATIONS samples with validation..."

python scripts/generate.py \
    --model_path "$MODEL_PATH" \
    --num_generations "$NUM_GENERATIONS" \
    --validate \
    --output_file "$OUTPUT_DIR/generations.txt" \
    --temperature 0.8 \
    --top_p 0.95 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/generation.log"

if [ $? -eq 0 ]; then
    print_status "✅ Generation completed"
else
    print_status "⚠️ Generation had issues"
fi

# =============================================================================
# 3. ANALYZE TRAINING LOGS
# =============================================================================
print_header "Step 3: Training Log Analysis"
print_status "Extracting training metrics..."

TRAINING_LOG="$HOME/training_success.log"

if [ -f "$TRAINING_LOG" ]; then
    # Extract loss values
    grep -E "'loss':|train_loss|eval_loss" "$TRAINING_LOG" > "$OUTPUT_DIR/training_metrics.txt" 2>/dev/null || true

    # Extract epoch summaries
    grep -E "epoch.*loss" "$TRAINING_LOG" | tail -20 > "$OUTPUT_DIR/epoch_summary.txt" 2>/dev/null || true

    # Count total steps
    TOTAL_STEPS=$(grep -E "[0-9]+/21882" "$TRAINING_LOG" | tail -1 | sed 's/.*\([0-9]\+\)\/21882.*/\1/' || echo "0")

    print_status "Total training steps: $TOTAL_STEPS"
fi

# =============================================================================
# 4. CREATE SUMMARY REPORT
# =============================================================================
print_header "Step 4: Creating Analysis Report"

cat > "$OUTPUT_DIR/ANALYSIS_REPORT.md" << 'EOFREPORT'
# Training Analysis Report
**Generated:** $(date)

## 📊 Model Information
- **Architecture:** GPT-2 Small (124M parameters)
- **Training Method:** LoRA (294K trainable parameters, 0.24%)
- **Dataset:** 700K samples (infix notation)
- **Training Duration:** $(grep "Training Duration:" $HOME/training_notification.txt 2>/dev/null | head -1 || echo "N/A")

## 📈 Training Metrics

### Loss Progression
```
$(tail -20 $OUTPUT_DIR/training_metrics.txt 2>/dev/null || echo "No metrics available")
```

### Epoch Summary
```
$(cat $OUTPUT_DIR/epoch_summary.txt 2>/dev/null || echo "No epoch data available")
```

## 🎯 Evaluation Results

### Performance Metrics
```
$(grep -E "Accuracy|Loss|Perplexity" $OUTPUT_DIR/evaluation.log 2>/dev/null || echo "Check evaluation.log for details")
```

### Sample Predictions
```
$(head -50 $OUTPUT_DIR/evaluation/*.txt 2>/dev/null | head -20 || echo "No evaluation samples found")
```

## 🔮 Generation Quality

### Validation Results
```
$(grep -E "Valid:|Success|Failed" $OUTPUT_DIR/generation.log | head -20 || echo "Check generation.log")
```

### Sample Generations
```
$(head -30 $OUTPUT_DIR/generations.txt 2>/dev/null || echo "No generations file found")
```

## 📁 Output Files
- Evaluation results: `evaluation/`
- Generated samples: `generations.txt`
- Full logs: `evaluation.log`, `generation.log`
- Training metrics: `training_metrics.txt`

## 🔗 Resources
- **Wandb Dashboard:** https://wandb.ai/symbolic-gression/seriguela_700K_test
- **HuggingFace Model:** https://huggingface.co/augustocsc/Se124M_700K_infix
- **Analysis Directory:** $OUTPUT_DIR

---
*Generated automatically by analyze_model.sh*
EOFREPORT

# Evaluate the report with actual values
eval "cat > \"$OUTPUT_DIR/ANALYSIS_REPORT.md\" << 'EOFREPORT'
$(cat "$OUTPUT_DIR/ANALYSIS_REPORT.md")
EOFREPORT"

print_status "Report created: $OUTPUT_DIR/ANALYSIS_REPORT.md"

# =============================================================================
# 5. FINAL SUMMARY
# =============================================================================
print_header "Analysis Complete!"
echo ""
print_status "All results saved to: $OUTPUT_DIR"
print_status "Main report: $OUTPUT_DIR/ANALYSIS_REPORT.md"
echo ""
print_status "Key files:"
echo "  - Evaluation: $OUTPUT_DIR/evaluation.log"
echo "  - Generation: $OUTPUT_DIR/generation.log"
echo "  - Metrics: $OUTPUT_DIR/training_metrics.txt"
echo "  - Report: $OUTPUT_DIR/ANALYSIS_REPORT.md"
echo ""
print_status "View the full report with:"
echo "  cat $OUTPUT_DIR/ANALYSIS_REPORT.md"
echo ""

# Create a quick summary
EVAL_SUCCESS=$(grep -c "✅" "$OUTPUT_DIR/evaluation.log" 2>/dev/null || echo "0")
GEN_SUCCESS=$(grep -c "Valid" "$OUTPUT_DIR/generation.log" 2>/dev/null || echo "0")

print_header "Quick Summary"
echo "Evaluation samples processed: $NUM_SAMPLES"
echo "Generations created: $NUM_GENERATIONS"
echo "Check logs for detailed metrics and quality assessment"
echo ""
print_status "Done!"
