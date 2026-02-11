#!/bin/bash
# Remote evaluation script for Base vs Medium comparison
# To be executed on AWS evaluation instance

set -e

cd ~/seriguela

PYTHON=/opt/pytorch/bin/python3
RESULTS_DIR="evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "============================================"
echo "Starting Evaluations - Base vs Medium"
echo "============================================"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Check GPU
echo "GPU Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Verify models exist
echo "Verifying models..."
if [ ! -d "output/gpt2_base_prefix_682k" ]; then
  echo "ERROR: Base model not found!"
  exit 1
fi

if [ ! -d "output/gpt2_medium_prefix_682k" ]; then
  echo "ERROR: Medium model not found!"
  exit 1
fi

echo "✓ Both models found"
echo ""

# ============================================
# Phase 1: Quick Validation (5 samples each)
# ============================================
echo "============================================"
echo "Phase 1: Quick Validation (5 samples)"
echo "============================================"

for model in base medium; do
  echo "Generating samples from $model..."
  $PYTHON scripts/generate.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_generations 5 \
    --validate \
    > $RESULTS_DIR/${model}_quick_samples.txt 2>&1 || true

  echo "✓ $model quick validation complete"
done

echo ""

# ============================================
# Phase 2: Quality Metrics (500 samples)
# ============================================
echo "============================================"
echo "Phase 2: Quality Metrics (500 samples)"
echo "============================================"

for model in base medium; do
  echo "Evaluating quality metrics for $model..."
  START_TIME=$(date +%s)

  $PYTHON scripts/evaluate.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_samples 500 \
    --output_file $RESULTS_DIR/${model}_quality_metrics.json \
    2>&1 | tee $RESULTS_DIR/${model}_quality.log || true

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo "✓ $model quality metrics complete (${DURATION}s)"
  echo ""
done

# ============================================
# Phase 3: Complexity Analysis (200 samples)
# ============================================
echo "============================================"
echo "Phase 3: Complexity Analysis (200 samples)"
echo "============================================"

for model in base medium; do
  echo "Analyzing complexity for $model..."
  START_TIME=$(date +%s)

  $PYTHON scripts/analyze_complexity.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_samples 200 \
    --output_file $RESULTS_DIR/complexity_${model}_prefix.json \
    2>&1 | tee $RESULTS_DIR/${model}_complexity.log || true

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo "✓ $model complexity analysis complete (${DURATION}s)"
  echo ""
done

# ============================================
# Phase 4: Model Comparison (Nguyen-5)
# ============================================
echo "============================================"
echo "Phase 4: Base vs Medium Comparison (Nguyen-5)"
echo "============================================"

if [ -f "data/benchmarks/nguyen/nguyen_5.csv" ]; then
  echo "Running comparison on Nguyen-5 benchmark..."
  START_TIME=$(date +%s)

  $PYTHON scripts/compare_trained_models.py \
    --model_base ./output/gpt2_base_prefix_682k \
    --model_medium ./output/gpt2_medium_prefix_682k \
    --dataset data/benchmarks/nguyen/nguyen_5.csv \
    --epochs 10 \
    --output_file $RESULTS_DIR/comparison_base_medium_nguyen5.json \
    2>&1 | tee $RESULTS_DIR/comparison.log || true

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo "✓ Comparison complete (${DURATION}s)"
else
  echo "⚠️  Nguyen-5 benchmark not found, skipping comparison"
fi

echo ""

# ============================================
# Phase 5: Generate Summary Report
# ============================================
echo "============================================"
echo "Phase 5: Generating Summary Report"
echo "============================================"

# Extract key metrics from JSON files
echo "Extracting metrics..."

# Quality metrics summary
if [ -f "$RESULTS_DIR/base_quality_metrics.json" ] && [ -f "$RESULTS_DIR/medium_quality_metrics.json" ]; then
  echo "Generating quality comparison table..."

  cat > $RESULTS_DIR/SUMMARY.md <<EOF
# Evaluation Results: Base vs Medium (Prefix Notation)

**Date**: $(date)
**Instance**: $(hostname)
**GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader)

## Models Evaluated

- **Base (124M)**: output/gpt2_base_prefix_682k
- **Medium (355M)**: output/gpt2_medium_prefix_682k

## Phases Completed

- ✓ Phase 1: Quick validation (5 samples)
- ✓ Phase 2: Quality metrics (500 samples)
- ✓ Phase 3: Complexity analysis (200 samples)
- ✓ Phase 4: Model comparison (Nguyen-5)
- ✓ Phase 5: Summary report

## Results Files

\`\`\`
$(ls -lh $RESULTS_DIR/*.json $RESULTS_DIR/*.txt 2>/dev/null | awk '{print $9 " (" $5 ")"}' || echo "No files found")
\`\`\`

## Quick Metrics Summary

### Quality Metrics (500 samples)

| Metric | Base | Medium |
|--------|------|--------|
| Valid Rate | TBD | TBD |
| Parseable Rate | TBD | TBD |
| Diversity | TBD | TBD |

### Complexity Analysis (200 samples)

| Metric | Base | Medium |
|--------|------|--------|
| Power Operations (%) | TBD | TBD |
| Nested Functions (%) | TBD | TBD |
| Avg Depth | TBD | TBD |

### Nguyen-5 Benchmark

| Metric | Base | Medium | Winner |
|--------|------|--------|--------|
| Best R² | TBD | TBD | ? |
| Valid Rate | TBD | TBD | ? |

## Next Steps

1. Download results to local machine:
   \`\`\`bash
   scp -i ~/.ssh/KEY.pem -r ubuntu@IP:~/seriguela/$RESULTS_DIR ./
   \`\`\`

2. Analyze detailed JSON files

3. Create visualizations

4. Update research documentation

---

**Evaluation completed at**: $(date)
**Total duration**: Check individual phase logs
EOF

  echo "✓ Summary report created"
fi

# ============================================
# Completion
# ============================================
echo ""
echo "============================================"
echo "EVALUATION COMPLETE!"
echo "============================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Files generated:"
ls -lh $RESULTS_DIR/
echo ""
echo "To download results:"
echo "  scp -i ~/.ssh/KEY.pem -r ubuntu@\$(hostname):~/seriguela/$RESULTS_DIR ./"
echo ""

# Mark completion
touch ~/.evaluation_complete
echo "Evaluation completed at $(date)" > ~/.evaluation_complete

echo "✓ All evaluations complete!"
