# Training Log: Model Scaling Experiment (Feb 2025)

## Experiment Metadata

- **Experiment Name**: Model Scaling for Symbolic Regression
- **Date Started**: 2026-02-02 23:41:37
- **Date Completed**: TBD (to be filled when training completes)
- **Total Duration**: TBD hours
- **Researcher**: Augusto César (augusto@example.com)
- **Git Commit**: e3e2787f1444f3690cd5d3c3300e0bb445c77216
- **Branch**: experiment/ppo-symbolic-regression

## Objective

Evaluate the impact of model size (124M → 355M → 774M parameters) on the ability to generate complex mathematical expressions for symbolic regression.

**Research Question**: Do larger models produce more complex, valid, and diverse expressions compared to smaller models?

**Hypotheses**:
1. Larger models will achieve higher valid expression rates
2. Larger models will generate deeper, more nested expressions
3. Larger models will use power operations (x²) more frequently
4. Larger models will achieve better R² scores on complex benchmarks (Nguyen-5+)

---

## Training Configuration

### Models Trained

| Model | Parameters | LoRA Trainable | Instance Type | Batch Size | Status |
|-------|-----------|----------------|---------------|-----------|--------|
| Base | 124M | 294K | g5.xlarge | 8 | 🟢 Training |
| Medium | 355M | 294K | g5.xlarge | 4 | 🟢 Training |
| Large | 774M | 294K | g5.2xlarge | 2 | 🟢 Training |

### Hyperparameters (Fixed Across All Models)

```json
{
  "dataset": "augustocsc/sintetico_natural",
  "data_subset": "700K",
  "data_column": "i_prompt_n",
  "format": "JSON (EXP-A)",
  "learning_rate": 5e-5,
  "num_train_epochs": 3,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 500,
  "weight_decay": 0.01,
  "early_stopping_patience": 3,
  "fp16": true,
  "seed": 42,
  "train_val_split": "90/10",
  "lora_r": 8,
  "lora_alpha": 32,
  "lora_target_modules": ["c_attn"],
  "lora_dropout": 0.05
}
```

**Rationale for fixed hyperparameters**: To isolate the effect of model size, all other variables must be held constant.

---

## AWS Infrastructure

### Instances Launched

| Model | Instance ID | Instance Type | IP Address | Launch Time | Training Started | Termination Time | Duration |
|-------|-------------|---------------|------------|-------------|------------------|------------------|----------|
| Base | i-0855711efcac25a9c | g5.xlarge | 18.234.96.235 | 2026-02-02 23:42:45 | 2026-02-03 10:13:00 | TBD | TBD |
| Medium | i-0eea77c3bbf1ea976 | g5.xlarge | 34.229.252.142 | 2026-02-02 23:43:00 | 2026-02-03 10:14:00 | TBD | TBD |
| Large | i-04dc6f51534d8185d | g5.2xlarge | 54.91.159.93 | 2026-02-02 23:43:16 | 2026-02-03 10:15:00 | TBD | TBD |

**Launch Method**: Parallel launch via `launch_all_models.sh` ✅

**AMI Used**: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)

**Security Group**: seriguela-sg

**SSH Key**: chave-gpu-nova

### Instance Status Verification

**All instances STOPPED**: ⏳ To be verified after training

**Verification command**:
```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-*-training" \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name]" \
  --output table
```

**Expected output after stopping**:
```
---------------------------------
|   DescribeInstances          |
+-------------+-----------------+
| i-xxxxx     | stopped         |
| i-xxxxx     | stopped         |
| i-xxxxx     | stopped         |
+-------------+-----------------+
```

### Costs

| Item | Hours | Rate (USD/h) | Cost (USD) |
|------|-------|--------------|------------|
| Base (g5.xlarge) | TBD | $1.006 | TBD |
| Medium (g5.xlarge) | TBD | $1.006 | TBD |
| Large (g5.2xlarge) | TBD | $1.212 | TBD |
| **Total Training** | TBD | - | **TBD** |

**Estimated Total**: $10-13 USD

**Actual Total**: TBD (to be filled after completion)

---

## Training Execution

### Pre-Training Checklist

- [ ] Git commit created and hash recorded
- [ ] Credentials loaded from `~/.tokens.txt`
- [ ] AWS CLI configured and working
- [ ] `train_with_json.py` tested locally
- [ ] Launch scripts corrected (cloud-init deadlock removed)
- [ ] All 3 launch scripts exist: base, medium, large

### Launch Log

**Launch command executed**:
```bash
bash launch_all_models.sh
```

**Launch time**: 2026-02-02 23:41:37

**Launch logs**:
- Base: `aws_launch_logs/launch_base.log`
- Medium: `aws_launch_logs/launch_medium.log`
- Large: `aws_launch_logs/launch_large.log`

**Launch results**:
- Base: ✅ Launched successfully (i-0855711efcac25a9c)
- Medium: ✅ Launched successfully (i-0eea77c3bbf1ea976)
- Large: ✅ Launched successfully (i-04dc6f51534d8185d)

### Training Progress

**Monitored via**: Weights & Biases dashboard

**Wandb Project**: seriguela

**Wandb Runs**:
- Base: TBD (seriguela-supervised-base-700k-TIMESTAMP)
- Medium: TBD (seriguela-supervised-medium-700k-TIMESTAMP)
- Large: TBD (seriguela-supervised-large-700k-TIMESTAMP)

**URLs**:
- Base: https://wandb.ai/YOUR_USERNAME/seriguela/runs/BASE_RUN_ID
- Medium: https://wandb.ai/YOUR_USERNAME/seriguela/runs/MEDIUM_RUN_ID
- Large: https://wandb.ai/YOUR_USERNAME/seriguela/runs/LARGE_RUN_ID

---

## Training Results

### Loss Curves

| Model | Initial Loss | Final Train Loss | Best Val Loss | Epoch Stopped | Early Stopping |
|-------|--------------|------------------|---------------|---------------|----------------|
| Base | TBD | TBD | TBD | TBD | ⏳ TBD |
| Medium | TBD | TBD | TBD | TBD | ⏳ TBD |
| Large | TBD | TBD | TBD | TBD | ⏳ TBD |

**Expected**: Lower loss for larger models due to increased capacity.

**Actual**: TBD

### Training Time

| Model | Estimated Time | Actual Time | Speedup/Slowdown |
|-------|----------------|-------------|------------------|
| Base | 2-3h | TBD | - |
| Medium | 3-4h | TBD | - |
| Large | 4-5h | TBD | - |

### Early Stopping Analysis

**Base**:
- Triggered: ⏳ TBD (Yes/No)
- Epoch stopped: TBD
- Reason: TBD (validation loss not improving / completed all epochs)

**Medium**:
- Triggered: ⏳ TBD
- Epoch stopped: TBD
- Reason: TBD

**Large**:
- Triggered: ⏳ TBD
- Epoch stopped: TBD
- Reason: TBD

---

## Model Outputs

### Files Generated

```
output/
├── gpt2_base_700K_json/
│   ├── adapter_model.bin (TBD MB)
│   ├── adapter_config.json
│   ├── config.json
│   ├── tokenizer_config.json
│   └── training_args.bin
├── gpt2_medium_700K_json/
│   └── [same structure] (TBD MB)
└── gpt2_large_700K_json/
    └── [same structure] (TBD MB)
```

**Model sizes**:
- Base adapter: TBD MB
- Medium adapter: TBD MB
- Large adapter: TBD MB

**Download commands**:
```bash
# Base
scp -i ~/.ssh/KEY.pem -r ubuntu@BASE_IP:~/seriguela/output/gpt2_base_700K_json ./output/

# Medium
scp -i ~/.ssh/KEY.pem -r ubuntu@MEDIUM_IP:~/seriguela/output/gpt2_medium_700K_json ./output/

# Large
scp -i ~/.ssh/KEY.pem -r ubuntu@LARGE_IP:~/seriguela/output/gpt2_large_700K_json ./output/
```

**Download status**:
- Base: ⏳ Pending
- Medium: ⏳ Pending
- Large: ⏳ Pending

---

## Issues Encountered

### Issue Log

#### Issue #1: Missing `train_with_json.py` in Git Repository
- **Severity**: High
- **When**: Initial Launch (2026-02-02 23:41:37)
- **Description**: The critical training script `train_with_json.py` was not committed to the Git repository. AWS instances cloned the repo but the script was missing, causing immediate training failure.
- **Error message**:
  ```
  can't open file '/home/ubuntu/seriguela/scripts/train_with_json.py': [Errno 2] No such file or directory
  ```
- **Resolution**:
  1. Stopped all instances
  2. Restarted instances (IPs changed: Base 18.234.96.235, Medium 34.229.252.142, Large 54.91.159.93)
  3. Uploaded script manually via SCP to all 3 instances
  4. Started training manually via SSH
- **Impact**:
  - **Time wasted**: ~13 hours of idle instances (overnight 2026-02-02 23:45 to 2026-02-03 10:00)
  - **Cost wasted**: ~$42 USD (3 instances × $1.00-1.20/h × 13h)
  - **Delay**: Training started 10.5 hours later than expected
- **Prevention**: Always commit critical scripts before launching AWS infrastructure. Add pre-flight check to verify script exists in repo.

#### Issue #2: TrainingArguments API Incompatibility
- **Severity**: High
- **When**: Training Initialization (2026-02-03 10:13:00)
- **Description**: After manual upload and restart, training failed during `TrainingArguments` initialization due to deprecated parameter name.
- **Error message**:
  ```
  TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
  File "/home/ubuntu/seriguela/scripts/train_with_json.py", line 154, in main
      training_args = TrainingArguments(
  ```
- **Resolution**:
  1. Updated `train_with_json.py` line 167: Changed `evaluation_strategy="steps"` to `eval_strategy="steps"`
  2. Re-uploaded corrected script to all 3 instances via SCP
  3. Killed failed training processes
  4. Restarted training successfully at 2026-02-03 10:13-10:15
- **Impact**:
  - **Time wasted**: ~10 minutes of failed training attempts
  - **Cost**: ~$0.50 USD
- **Prevention**: Test training script on AWS environment before mass deployment. Keep transformers library documentation updated.

#### Issue #3: Automatic Monitoring Required
- **Severity**: Medium
- **When**: After Issue #1 discovery (2026-02-03 10:00)
- **Description**: User requested automatic monitoring after discovering instances ran idle overnight without doing anything.
- **Resolution**:
  1. Created `monitor_training.sh` - checks every 5 minutes for completion
  2. Created `quick_check.sh` - manual quick status check
  3. Monitor automatically downloads models and stops instances when all training completes
  4. Started monitoring at 2026-02-03 10:23:41
- **Impact**: Prevention of future idle time and cost waste
- **Prevention**: Always include monitoring infrastructure when launching long-running AWS jobs

**Total Issues**: 3 (2 High, 1 Medium)
**Total Time Lost**: ~13.5 hours
**Total Cost Waste**: ~$42.50 USD

---

## Evaluation Summary

*To be filled after evaluation phase completes*

### Basic Quality Metrics

| Metric | Base | Medium | Large | Best Model |
|--------|------|--------|-------|-----------|
| Valid Expression Rate (%) | TBD | TBD | TBD | TBD |
| Parseable Rate (%) | TBD | TBD | TBD | TBD |
| Constraint Adherence (%) | TBD | TBD | TBD | TBD |
| Diversity Rate (%) | TBD | TBD | TBD | TBD |

### Complexity Metrics

| Metric | Base | Medium | Large | Improvement (B→L) |
|--------|------|--------|-------|-------------------|
| Power Operations (%) | TBD | TBD | TBD | TBD |
| Nested Trig (%) | TBD | TBD | TBD | TBD |
| Average Depth | TBD | TBD | TBD | TBD |

### Nguyen Benchmark Performance

**Average R² across all 12 benchmarks**:
- Base: TBD
- Medium: TBD
- Large: TBD

**Best algorithm by model**:
- Base: TBD (REINFORCE/GRPO/PPO)
- Medium: TBD
- Large: TBD

*See `EXPERIMENT_MODEL_SCALING.md` for detailed analysis*

---

## Checklist

### Training Phase
- [ ] All 3 models trained successfully
- [ ] No errors in training logs
- [ ] Wandb tracking enabled and working
- [ ] Early stopping worked correctly
- [ ] All models saved to disk
- [ ] Training logs downloaded from instances

### AWS Cleanup
- [ ] All instances verified stopped
- [ ] Final costs calculated and documented
- [ ] No orphaned volumes or snapshots
- [ ] Instance info files saved locally

### Documentation
- [ ] This training log completed
- [ ] Model cards created (base, medium, large)
- [ ] EXPERIMENT_MODEL_SCALING.md created
- [ ] CLAUDE.md updated with results
- [ ] Git commit with all documentation

### HuggingFace Publication
- [ ] Base model uploaded to HuggingFace
- [ ] Medium model uploaded to HuggingFace
- [ ] Large model uploaded to HuggingFace
- [ ] Model cards published
- [ ] Repository links added to documentation

### Evaluation Phase
- [ ] Basic quality metrics collected (all 3 models)
- [ ] Complexity analysis completed (all 3 models)
- [ ] Nguyen suite (144 experiments) completed
- [ ] Results aggregated and visualized
- [ ] Statistical analysis performed
- [ ] Key findings documented

---

## Final Notes

### Success Criteria

**Minimum viable**:
- ✅/❌ All 3 models trained without errors
- ✅/❌ Basic metrics collected for comparison
- ✅/❌ Results documented

**Complete success**:
- ✅/❌ Measurable improvement with model size
- ✅/❌ Statistical significance demonstrated
- ✅/❌ Publication-ready report with visualizations
- ✅/❌ All models published to HuggingFace

### Key Learnings

*To be filled after experiment completes*

1. **Model scaling impact**: TBD
2. **Optimal model size**: TBD
3. **Cost-benefit analysis**: TBD
4. **Unexpected findings**: TBD

### Future Work

*To be filled after experiment completes*

1. TBD
2. TBD
3. TBD

---

**Log Started**: 2026-02-02 23:41:37

**Last Updated**: 2026-02-03 10:25:00

**Status**: 🟢 Training In Progress (All 3 models running successfully)

**Monitoring**: ✅ Active (automatic check every 5 minutes)
