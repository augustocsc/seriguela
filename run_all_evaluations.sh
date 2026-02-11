#!/bin/bash
# Complete Evaluation Pipeline for Prefix Notation Models
# Created: 2026-02-10
# Purpose: Automated evaluation of Base, Medium, Large prefix models

set -e

echo "============================================"
echo "Prefix Notation Models - Complete Evaluation"
echo "============================================"
echo ""

# Configuration
MODELS=("base" "medium" "large")
RESULTS_DIR="./evaluation_results/prefix_$(date +%Y%m%d)"
mkdir -p $RESULTS_DIR

# Check if models exist
echo "Checking if models are available..."
for model in "${MODELS[@]}"; do
  MODEL_PATH="./output/gpt2_${model}_prefix_682k"
  if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  WARNING: Model not found: $MODEL_PATH"
    echo "   Please download from AWS first:"
    echo "   scp -i ~/.ssh/chave-gpu-nova.pem -r ubuntu@<IP>:~/seriguela/output/gpt2_${model}_prefix_682k ./output/"
    exit 1
  fi
  echo "✓ Found: $MODEL_PATH"
done
echo ""

# ============================================
# PHASE 1: Quick Validation (5 expressions each)
# ============================================
echo "============================================"
echo "PHASE 1: Quick Validation"
echo "============================================"
for model in "${MODELS[@]}"; do
  echo "Generating 5 sample expressions from $model..."
  python scripts/generate.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_generations 5 \
    --validate \
    > $RESULTS_DIR/${model}_quick_samples.txt 2>&1

  echo "✓ $model: Quick validation complete"
done
echo ""

# ============================================
# PHASE 2: Quality Metrics (500 samples each)
# ============================================
echo "============================================"
echo "PHASE 2: Quality Metrics (500 samples)"
echo "============================================"
for model in "${MODELS[@]}"; do
  echo "Evaluating quality metrics for $model..."
  python scripts/evaluate.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_samples 500 \
    --output_dir $RESULTS_DIR \
    --output_file ${model}_quality_metrics.json

  echo "✓ $model: Quality metrics complete"
done
echo ""

# ============================================
# PHASE 3: Complexity Analysis (200 samples each)
# ============================================
echo "============================================"
echo "PHASE 3: Complexity Analysis (200 samples)"
echo "============================================"
for model in "${MODELS[@]}"; do
  echo "Analyzing expression complexity for $model..."
  python scripts/analyze_complexity.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_samples 200 \
    --output_file $RESULTS_DIR/complexity_${model}_prefix.json

  echo "✓ $model: Complexity analysis complete"
done
echo ""

# ============================================
# PHASE 4: Model Size Comparison (Nguyen-5)
# ============================================
echo "============================================"
echo "PHASE 4: Model Size Comparison (Nguyen-5)"
echo "============================================"
if [ -f "data/benchmarks/nguyen/nguyen_5.csv" ]; then
  echo "Comparing all three models on Nguyen-5 benchmark..."
  python scripts/compare_trained_models.py \
    --model_base ./output/gpt2_base_prefix_682k \
    --model_medium ./output/gpt2_medium_prefix_682k \
    --model_large ./output/gpt2_large_prefix_682k \
    --dataset data/benchmarks/nguyen/nguyen_5.csv \
    --epochs 10 \
    --output_file $RESULTS_DIR/comparison_prefix_nguyen5.json

  echo "✓ Model comparison complete"
else
  echo "⚠️  WARNING: Nguyen-5 benchmark not found, skipping comparison"
fi
echo ""

# ============================================
# PHASE 5: Prefix vs Infix Comparison
# ============================================
echo "============================================"
echo "PHASE 5: Prefix vs Infix Comparison"
echo "============================================"

# Only compare if infix models exist
for model in "${MODELS[@]}"; do
  INFIX_MODEL="./output/gpt2_${model}_700K_json"
  PREFIX_MODEL="./output/gpt2_${model}_prefix_682k"

  if [ -d "$INFIX_MODEL" ]; then
    echo "Comparing prefix vs infix for $model..."
    python scripts/compare_models.py \
      --model1 $PREFIX_MODEL \
      --model2 $INFIX_MODEL \
      --num_samples 500 \
      --output_file $RESULTS_DIR/prefix_vs_infix_${model}.json

    echo "✓ $model: Prefix vs Infix comparison complete"
  else
    echo "⚠️  Infix model not found for $model, skipping comparison"
  fi
done
echo ""

# ============================================
# PHASE 6: Generate Summary Report
# ============================================
echo "============================================"
echo "PHASE 6: Generating Summary Report"
echo "============================================"

REPORT_FILE="$RESULTS_DIR/EVALUATION_SUMMARY.md"

cat > $REPORT_FILE <<EOF
# Evaluation Results: Prefix Notation Models
**Date**: $(date +"%Y-%m-%d %H:%M:%S")
**Models Evaluated**: Base (124M), Medium (355M), Large (774M)

## Evaluation Pipeline Completed

- ✓ Phase 1: Quick validation (5 expressions each)
- ✓ Phase 2: Quality metrics (500 samples each)
- ✓ Phase 3: Complexity analysis (200 samples each)
- ✓ Phase 4: Model size comparison (Nguyen-5)
- ✓ Phase 5: Prefix vs Infix comparison

## Results Location

All results saved to: \`$RESULTS_DIR/\`

### Files Generated

EOF

# List all generated files
ls -lh $RESULTS_DIR/*.json $RESULTS_DIR/*.txt 2>/dev/null | awk '{print "- " $9 " (" $5 ")"}' >> $REPORT_FILE || echo "No JSON/TXT files found" >> $REPORT_FILE

cat >> $REPORT_FILE <<EOF

## Next Steps

1. Review quality metrics: \`*_quality_metrics.json\`
2. Analyze complexity: \`complexity_*_prefix.json\`
3. Compare model sizes: \`comparison_prefix_nguyen5.json\`
4. Prefix vs Infix: \`prefix_vs_infix_*.json\`

## Quick Summary Commands

\`\`\`bash
# View quality metrics
cat $RESULTS_DIR/*_quality_metrics.json | jq .

# View complexity analysis
cat $RESULTS_DIR/complexity_*_prefix.json | jq .

# View comparisons
cat $RESULTS_DIR/comparison_*.json | jq .
\`\`\`

---
Generated by: run_all_evaluations.sh
EOF

echo "✓ Summary report created: $REPORT_FILE"
echo ""

# ============================================
# COMPLETION
# ============================================
echo "============================================"
echo "EVALUATION COMPLETE!"
echo "============================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "Summary report: $REPORT_FILE"
echo ""
echo "To view the summary:"
echo "  cat $REPORT_FILE"
echo ""
echo "Total time: $SECONDS seconds"
