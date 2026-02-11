# Evaluation Guide: Prefix Notation Models

**Created**: 2026-02-10
**Models**: GPT-2 Base (124M), Medium (355M), Large (774M)
**Dataset**: augustocsc/sintetico_natural_prefix_682k

## Training Status

| Model | Instance ID | Status | Location |
|-------|-------------|--------|----------|
| Base (124M) | i-03cb806bdc98e6d36 | ✅ COMPLETE | 3.233.238.126 |
| Medium (355M) | i-0567ed93f9e625a89 | ✅ COMPLETE | 100.52.210.14 |
| Large (774M) | i-060e3e00d1138c964 | ⏳ TRAINING | 18.206.201.220 |

**Action Taken**: Base and Medium instances STOPPED to save costs (~$2/hour).

## Quick Start: Download Models

```bash
# Monitor LARGE training progress
bash monitor_large_training.sh

# Download all completed models and stop instances
bash download_and_stop_models.sh
```

## Evaluation Pipeline

Once all models are downloaded to `./output/`, run:

```bash
# Run complete evaluation pipeline
bash run_all_evaluations.sh
```

This will execute:
1. **Quick validation**: Generate 5 sample expressions per model
2. **Quality metrics**: Evaluate 500 expressions per model
   - Valid expression rate
   - Parseable rate
   - Constraint adherence
   - Diversity rate
3. **Complexity analysis**: Analyze 200 expressions per model
   - Power operations usage
   - Nested functions
   - Expression depth
4. **Model comparison**: Compare Base vs Medium vs Large on Nguyen-5
5. **Prefix vs Infix**: Compare prefix notation vs infix notation models

**Estimated time**: 3-5 hours (depending on GPU)

**Output**: All results saved to `./evaluation_results/prefix_YYYYMMDD/`

## Manual Steps

### 1. Monitor LARGE Training

```bash
# Check status
bash monitor_large_training.sh

# Or manually SSH
ssh -i C:/Users/madeinweb/chave-gpu.pem ubuntu@18.206.201.220
tail -f ~/seriguela/training_large_prefix.log
```

### 2. Download Models

```bash
# Download Base (if not already done)
scp -i C:/Users/madeinweb/chave-gpu.pem -r \
  ubuntu@3.233.238.126:~/seriguela/output/gpt2_base_prefix_682k \
  ./output/

# Download Medium (if not already done)
scp -i C:/Users/madeinweb/chave-gpu.pem -r \
  ubuntu@100.52.210.14:~/seriguela/output/gpt2_medium_prefix_682k \
  ./output/

# Download Large (when complete)
scp -i C:/Users/madeinweb/chave-gpu.pem -r \
  ubuntu@18.206.201.220:~/seriguela/output/gpt2_large_prefix_682k \
  ./output/
```

### 3. Stop Instances (IMPORTANT!)

```bash
# Stop all prefix training instances
aws ec2 stop-instances --instance-ids \
  i-03cb806bdc98e6d36 \
  i-0567ed93f9e625a89 \
  i-060e3e00d1138c964
```

### 4. Run Individual Evaluations

```bash
# Quick test (5 generations)
python scripts/generate.py \
  --model_path ./output/gpt2_base_prefix_682k \
  --num_generations 5 \
  --validate

# Quality metrics (500 samples)
python scripts/evaluate.py \
  --model_path ./output/gpt2_base_prefix_682k \
  --num_samples 500 \
  --output_file ./evaluation_results/base_quality.json

# Complexity analysis (200 samples)
python scripts/analyze_complexity.py \
  --model_path ./output/gpt2_base_prefix_682k \
  --num_samples 200 \
  --output_file ./evaluation_results/complexity_base.json

# Compare all three sizes
python scripts/compare_trained_models.py \
  --model_base ./output/gpt2_base_prefix_682k \
  --model_medium ./output/gpt2_medium_prefix_682k \
  --model_large ./output/gpt2_large_prefix_682k \
  --dataset data/benchmarks/nguyen/nguyen_5.csv \
  --epochs 10

# Compare prefix vs infix
python scripts/compare_models.py \
  --model1 ./output/gpt2_base_prefix_682k \
  --model2 ./output/gpt2_base_700K_json \
  --num_samples 500
```

## Expected Results

### Hypothesis 1: Valid Expression Rate
- **Base**: 80%+ (similar to infix)
- **Medium**: 85%+
- **Large**: 90%+

### Hypothesis 2: Complexity
- **Base**: Depth ~1.5, Power ops 15-20%
- **Medium**: Depth ~2.0, Power ops 30-40%
- **Large**: Depth ~2.5+, Power ops 50%+, Nested functions >0%

### Hypothesis 3: Prefix vs Infix
- **Prefix**: Higher parseable rate (more structured)
- **Infix**: Similar performance, more human-readable
- **Recommendation**: Choose based on downstream application

## Evaluation Results

After running evaluations, results will be in:
```
evaluation_results/prefix_YYYYMMDD/
├── base_quality_metrics.json
├── medium_quality_metrics.json
├── large_quality_metrics.json
├── complexity_base_prefix.json
├── complexity_medium_prefix.json
├── complexity_large_prefix.json
├── comparison_prefix_nguyen5.json
├── prefix_vs_infix_base.json
├── prefix_vs_infix_medium.json
├── prefix_vs_infix_large.json
└── EVALUATION_SUMMARY.md
```

## Cost Summary

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| Base training | ~3.5h | $1.006/h | ~$3.52 |
| Medium training | ~8.5h | $1.006/h | ~$8.55 |
| Large training | ~24h | $1.212/h | ~$29.09 |
| **TOTAL TRAINING** | | | **~$41.16** |
| Evaluation (local GPU) | - | $0 | $0 |
| Evaluation (AWS g5.xlarge) | ~5h | $1.006/h | ~$5.03 |

## Checklist

**Before Evaluation**:
- [x] Base training complete
- [x] Medium training complete
- [ ] Large training complete
- [x] Base instance STOPPED
- [x] Medium instance STOPPED
- [ ] Large instance STOPPED (after download)

**Download**:
- [ ] Base model downloaded to `./output/gpt2_base_prefix_682k`
- [ ] Medium model downloaded to `./output/gpt2_medium_prefix_682k`
- [ ] Large model downloaded to `./output/gpt2_large_prefix_682k`

**Evaluation**:
- [ ] Quick validation passed
- [ ] Quality metrics collected
- [ ] Complexity analysis done
- [ ] Model comparison complete
- [ ] Prefix vs Infix comparison done

**Documentation**:
- [ ] Results summarized in `EVALUATION_SUMMARY.md`
- [ ] Update `CLAUDE.md` with findings
- [ ] Create final report for article

## Troubleshooting

### Models not found
```bash
# Check if models exist
ls -lh ./output/gpt2_*_prefix_682k
```

### SSH connection issues
```bash
# Test SSH connection
ssh -i C:/Users/madeinweb/chave-gpu.pem ubuntu@3.233.238.126 'echo "Connection OK"'

# If permission denied, check key permissions
chmod 400 C:/Users/madeinweb/chave-gpu.pem
```

### Evaluation taking too long
```bash
# Reduce sample sizes
python scripts/evaluate.py --num_samples 100  # instead of 500
python scripts/analyze_complexity.py --num_samples 50  # instead of 200
```

### Out of memory during evaluation
```bash
# Use smaller batch sizes (modify script)
# Or run on AWS g5.xlarge instance
```

## Next Steps After Evaluation

1. **Analyze results**: Review JSON files and summary
2. **Create visualizations**: Plot comparisons (Base vs Medium vs Large)
3. **Statistical analysis**: Test significance of differences
4. **Write findings**: Update research paper/article
5. **Model cards**: Create HuggingFace model cards
6. **Publication**: Upload models to HuggingFace Hub

## Contact

For questions or issues, see `CLAUDE.md` for project context.

---
**Last Updated**: 2026-02-10
**Status**: LARGE model still training, Base and Medium complete
