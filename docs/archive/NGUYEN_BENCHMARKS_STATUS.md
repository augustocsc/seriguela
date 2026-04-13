# Nguyen Benchmarks Evaluation - Status

**Date**: 2026-02-04
**Status**: 🔄 **IN PROGRESS** (2/36 experiments completed)
**Instance**: i-07279c5889587abfe (54.91.123.196)

---

## 🎯 Experiment Details

**Objective**: Evaluate 3 models (Base, Medium, Large) on 12 Nguyen benchmarks with R² scoring

**Configuration**:
- **Models**: Base (124M), Medium (355M), Large (774M)
- **Benchmarks**: Nguyen 1-12
- **Total experiments**: 36 (3 models × 12 benchmarks)
- **Samples per experiment**: 100 candidate expressions
- **Total generations**: 3,600 expressions

**AWS Infrastructure**:
- **Instance Type**: g5.xlarge (NVIDIA A10G GPU, 24GB VRAM)
- **Instance ID**: i-07279c5889587abfe
- **Public IP**: 54.91.123.196
- **Estimated time**: 2-3 hours
- **Estimated cost**: ~$3-4 USD

---

## 📊 Progress (Real-Time)

### Completed Experiments (2/36)

| # | Model | Benchmark | Valid Rate | Best R² | Duration |
|---|-------|-----------|------------|---------|----------|
| 1 | Base | Nguyen-1 | 49% (49/100) | **0.9717** 🏆 | 97s |
| 2 | Base | Nguyen-2 | 43% (43/100) | **0.9110** | 98s |

### Current Status

- **Experiments completed**: 2 / 36 (5.6%)
- **Current**: Base + Nguyen-6 (running)
- **Time elapsed**: ~2 minutes
- **Estimated remaining**: ~2h50min

**Key Findings So Far**:
- ✅ Best R² = 0.9717 (97.17% fit!) on Nguyen-1
- ✅ Valid expression rates: 43-49%
- ✅ System working perfectly on AWS

---

## 🔍 Monitoring Commands

### Check Progress

```bash
bash monitor_nguyen.sh
```

### Watch Live Log

```bash
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.91.123.196 'tail -f ~/seriguela/nguyen_benchmarks.log'
```

### Check GPU Usage

```bash
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.91.123.196 'nvidia-smi'
```

### Check if Complete

```bash
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.91.123.196 'grep "SUITE COMPLETE" ~/seriguela/nguyen_benchmarks.log'
```

---

## 📥 Download Results (When Complete)

### 1. Verify Completion

```bash
bash monitor_nguyen.sh
```

Look for "ALL EXPERIMENTS COMPLETE!" message.

### 2. Download Results

```bash
scp -i "C:/Users/madeinweb/chave-gpu.pem" -r ubuntu@54.91.123.196:~/seriguela/results_nguyen_benchmarks ./
```

### 3. Stop Instance **IMMEDIATELY**

```bash
aws ec2 stop-instances --instance-ids i-07279c5889587abfe
```

**⚠️ CRITICAL**: Stop the instance to avoid additional costs!

### 4. Verify Instance Stopped

```bash
aws ec2 describe-instances --instance-ids i-07279c5889587abfe --query 'Reservations[0].Instances[0].State.Name' --output text
```

Should return: `stopped`

---

## 📁 Expected Results Structure

After download, you should have:

```
results_nguyen_benchmarks/
├── summary.json                    # Overall summary
├── base_nguyen1.json              # Base + Nguyen-1 results
├── base_nguyen2.json              # Base + Nguyen-2 results
├── ...                            # (36 result files total)
├── medium_nguyen1.json
├── ...
├── large_nguyen1.json
└── ...
```

**Each result file contains**:
- All 100 generated expressions
- R² score for each valid expression
- Best R² achieved
- Mean/median R² statistics
- Valid expression rate
- Error analysis

---

## 📈 Next Steps (After Download)

### 1. Aggregate Results

```bash
python scripts/aggregate_nguyen_results.py \
  --input_dir results_nguyen_benchmarks \
  --output_dir nguyen_analysis
```

### 2. Update Scientific Report

Add benchmark results to:
- `SCIENTIFIC_REPORT_MODEL_SCALING.md`
- Create new section: "Nguyen Benchmark Performance"
- Include R² comparisons across models
- Generate visualizations (heatmaps, bar charts)

### 3. Analysis Focus

**Key Questions to Answer**:
1. Do larger models achieve better R² scores?
2. Which benchmarks are easier/harder for each model?
3. Is there a correlation between model size and R² improvement?
4. What types of expressions do larger models generate?

**Statistical Tests**:
- ANOVA: Compare R² scores across models
- Paired t-tests: Base vs Medium, Medium vs Large
- Effect sizes (Cohen's d)
- Confidence intervals

---

## 💰 Cost Tracking

### Training Phase (Completed)
- Base: $2-3
- Medium: $3-4
- Large: $5-6
- **Training Total**: $10-13

### Evaluation Phase
- Quality eval: $2.50 (completed)
- Nguyen benchmarks: $3-4 (in progress)
- **Evaluation Total**: $5.50-6.50

### **Grand Total**: $15.50-19.50 USD

---

## 🚨 Troubleshooting

### If Execution Fails

**Check if process crashed**:
```bash
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.91.123.196 'ps aux | grep python'
```

**Check error logs**:
```bash
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.91.123.196 'tail -100 ~/seriguela/nguyen_benchmarks.log | grep -i error'
```

**Restart manually if needed**:
```bash
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.91.123.196
cd ~/seriguela
nohup python3 scripts/run_all_nguyen_benchmarks.py \
  --models base medium large \
  --benchmarks 1 2 3 4 5 6 7 8 9 10 11 12 \
  --num_samples 100 \
  --output_dir ./results_nguyen_benchmarks \
  --models_dir ./output \
  > nguyen_benchmarks_restart.log 2>&1 &
```

### If Connection Times Out

**Check instance status**:
```bash
aws ec2 describe-instances --instance-ids i-07279c5889587abfe --query 'Reservations[0].Instances[0].State.Name' --output text
```

**Restart if stopped**:
```bash
aws ec2 start-instances --instance-ids i-07279c5889587abfe
```

---

## 📞 Quick Reference

**Instance ID**: i-07279c5889587abfe
**Public IP**: 54.91.123.196
**Key**: C:/Users/madeinweb/chave-gpu.pem

**Monitor**: `bash monitor_nguyen.sh`
**Download**: See "Download Results" section above
**Stop**: `aws ec2 stop-instances --instance-ids i-07279c5889587abfe`

---

**Document Updated**: 2026-02-04 10:01
**Next Update**: After completion (estimated ~13:00-13:30)
**Status Check**: Run `bash monitor_nguyen.sh` anytime
