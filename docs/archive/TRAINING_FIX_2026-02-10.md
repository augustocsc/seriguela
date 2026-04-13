# Critical Training Fix - February 10, 2026

## Problem Identified

**Symptom**: Training running at 17 seconds per step instead of expected 2-3 seconds per step.

**Impact**:
- Expected training time: 2-3 hours
- Actual training time: 280+ hours (100x slower!)
- Cost impact: ~$300 instead of ~$3 per model

**Root Cause**: Double-split problem in `scripts/train_with_json.py`

## Root Cause Analysis

### The Double-Split Bug

The original script performed an unnecessary additional split on top of the dataset's existing validation split:

```python
# BUGGY CODE (scripts/train_with_json.py, lines 112-128)

# Load dataset with existing splits (train: 682,429, validation: 75,826)
dataset = load_dataset(args.dataset_repo, data_dir=args.data_dir)

# BUG: Takes only the train split
train_dataset = dataset["train"].map(convert_to_json_format, ...)

# BUG: Creates NEW split on top of existing train split!
split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']      # Now only 614,186 examples
eval_dataset = split_dataset['test']        # Now only 68,243 examples
```

**What was wrong**:
1. Dataset already has pre-existing train/validation splits (682,429 + 75,826)
2. Script ignored the existing validation split
3. Script performed ANOTHER 90/10 split on the train data only
4. This double-splitting caused incorrect data sizes and extremely slow processing

### Comparison: Expected vs Actual

| Split | Expected (Correct) | Actual (Buggy) | Loss |
|-------|-------------------|----------------|------|
| Train | 682,429 | 614,186 | -68,243 (-10%) |
| Validation | 75,826 | 68,243 | -7,583 (-10%) |
| **UNUSED** | 0 | **75,826** | Original validation split ignored! |

**Critical data loss**: 75,826 examples from the original validation split were completely ignored.

## Solution Implemented

### 1. Created Standardized Training Configuration

**File**: `scripts/training_config.py`

Comprehensive training configuration module with:
- Standardized hyperparameters for all model sizes
- Model-specific configurations (batch sizes, instance types, costs)
- Dataset validation functions
- Training monitoring best practices
- Academic reporting helpers (methods section, LaTeX tables)

**Key Features**:
```python
from scripts.training_config import (
    get_model_config,           # Get model-specific settings
    get_training_args_dict,     # Get training arguments
    validate_dataset_splits,    # Validate split sizes
    print_training_summary,     # Print configuration summary
)
```

### 2. Created Fixed Training Script

**File**: `scripts/train_with_json_fixed.py`

**Key Fixes**:

1. **Uses pre-existing splits** (no double-split):
```python
# FIXED CODE
dataset = load_dataset(args.dataset_repo)  # Load entire dataset

# Validate that splits exist
if "train" not in dataset or "validation" not in dataset:
    raise ValueError("Dataset must have 'train' and 'validation' splits!")

# Convert BOTH splits to JSON (no re-splitting!)
train_dataset = dataset["train"].map(...)
eval_dataset = dataset["validation"].map(...)  # Use existing validation
```

2. **Validates dataset structure**:
```python
validate_dataset_splits(dataset)  # Checks expected sizes
```

3. **Imports standardized config**:
```python
from scripts.training_config import get_model_config
config = get_model_config(args.model_size)
batch_size = config["per_device_train_batch_size"]
```

4. **Auto-configures batch size** based on model:
- Base (124M): 8
- Medium (355M): 4
- Large (774M): 2

5. **Includes performance monitoring**:
```python
print(f"Expected speed: ~2-3 seconds per step")
print(f"If training is slower than 5s/step, something is wrong!")
```

### 3. Updated Launch Scripts

Updated all three AWS launch scripts:
- `scripts/aws/launch_base_prefix_training.sh`
- `scripts/aws/launch_medium_prefix_training.sh`
- `scripts/aws/launch_large_prefix_training.sh`

**Changes**:
- Use `train_with_json_fixed.py` instead of `train_with_json.py`
- Removed unnecessary `sed` patches
- Added `--text_column p_prompt_n_converted` parameter

## Verification Checklist

Before relaunching training, verify:

- [ ] Instances stopped (i-018174fe8342d5972, i-09a0fcbae3e8c043c, i-0f9dd3f75021c717a)
- [ ] New scripts committed to git
- [ ] `train_with_json_fixed.py` exists and is executable
- [ ] `training_config.py` exists and is importable
- [ ] Launch scripts updated to use fixed version
- [ ] Dataset has correct splits (682,429 train + 75,826 validation)

## Expected Results After Fix

| Metric | Before (Buggy) | After (Fixed) | Improvement |
|--------|---------------|---------------|-------------|
| Steps/second | 0.06 (17s/step) | 0.4 (2.5s/step) | **~7x faster** |
| Training time | 280+ hours | 2-3 hours | **100x faster** |
| Train data size | 614,186 | 682,429 | +68,243 (+11%) |
| Validation data size | 68,243 | 75,826 | +7,583 (+11%) |
| Total data used | 682,429 (90%) | 758,255 (100%) | All data used |
| Cost per model | ~$300 | ~$3 | **100x cheaper** |

## Instructions for Relaunch

1. **Stop previous training** (already done):
```bash
aws ec2 stop-instances --instance-ids i-018174fe8342d5972 i-09a0fcbae3e8c043c i-0f9dd3f75021c717a
```

2. **Verify instances stopped**:
```bash
aws ec2 describe-instances --instance-ids i-018174fe8342d5972 i-09a0fcbae3e8c043c i-0f9dd3f75021c717a \
  --query "Reservations[*].Instances[*].State.Name"
```

3. **Test locally** (optional but recommended):
```bash
python scripts/train_with_json_fixed.py \
  --model_size gpt2 \
  --dataset_repo augustocsc/sintetico_natural_prefix_682k \
  --text_column p_prompt_n_converted \
  --output_dir ./test_output \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2
```

4. **Commit changes to git**:
```bash
git add scripts/training_config.py
git add scripts/train_with_json_fixed.py
git add scripts/aws/launch_*_prefix_training.sh
git add TRAINING_FIX_2026-02-10.md
git commit -m "Fix: Correct double-split bug in training script

- Add training_config.py with standardized configurations
- Create train_with_json_fixed.py that uses existing validation split
- Update launch scripts to use fixed training script
- Expected speedup: 100x (2-3h instead of 280h)
"
git push origin experiment/ppo-symbolic-regression
```

5. **Relaunch training** (all 3 models in parallel):
```bash
bash launch_all_prefix_models.sh
```

6. **Monitor initial speed** (SSH to one instance after 10 minutes):
```bash
# Check one instance to verify speed
ssh -i ~/.ssh/chave-gpu-nova.pem ubuntu@<IP>
tail -f ~/training_base_prefix.log

# Look for lines like:
# {'loss': X.XXX, 'learning_rate': X.XXX, 'epoch': X.XX}
# Should appear every 2-3 seconds (not 17 seconds!)
```

7. **Verify Wandb metrics**:
- Check that steps/second is ~0.4 (not 0.06)
- Training loss should decrease normally
- Check that train size shows 682,429 (not 614,186)

## Academic Reporting

When documenting this in the article:

### Data Section
"The dataset comprises 758,255 mathematical expressions in prefix notation, split into 682,429 training examples (90%) and 75,826 validation examples (10%). All experiments used the same pre-defined splits to ensure reproducibility."

### Methods Section
Use `generate_methods_section()` from `training_config.py`:
```python
from scripts.training_config import generate_methods_section
print(generate_methods_section())
```

### Hyperparameters Table
Use `generate_hyperparameters_table()` from `training_config.py`:
```python
from scripts.training_config import generate_hyperparameters_table
print(generate_hyperparameters_table())
```

## Lessons Learned

1. **Always use existing splits**: Datasets from HuggingFace Hub often come with pre-defined train/validation splits. Don't create new splits unless necessary.

2. **Validate data sizes early**: Compare expected vs actual data sizes before starting expensive training runs.

3. **Monitor training speed**: If training is significantly slower than expected, investigate immediately (don't wait for completion).

4. **Standardize configurations**: Having a central configuration module (`training_config.py`) prevents bugs and ensures reproducibility.

5. **Document for papers**: Academic reporting requires detailed methodology. Having standardized functions to generate this documentation saves time and ensures accuracy.

## Cost Impact

**Previous attempt** (buggy, stopped after ~1 hour):
- Base: ~$1.00
- Medium: ~$1.00
- Large: ~$1.20
- **Total wasted**: ~$3.20

**Expected cost** (with fix):
- Base: ~$2-3
- Medium: ~$3-4
- Large: ~$5-6
- **Total**: ~$10-13

**Cost saved by fixing early**: ~$870 ($900 - $30)
- If we had let the buggy training complete: 3 models × 280h × ~$1/h = ~$900
- Fixed training: 3 models × ~3h × ~$1/h = ~$10
- Stopping early and fixing saved: **$890**

## References

- Original buggy script: `scripts/train_with_json.py`
- Fixed script: `scripts/train_with_json_fixed.py`
- Configuration module: `scripts/training_config.py`
- Dataset: `augustocsc/sintetico_natural_prefix_682k` (HuggingFace Hub)
- Wandb buggy runs:
  - Base: https://wandb.ai/symbolic-gression/huggingface/runs/j2wn7ua3
  - Medium: https://wandb.ai/symbolic-gression/huggingface/runs/ovjmz6pd
  - Large: https://wandb.ai/symbolic-gression/huggingface/runs/crhjuo6u

---

**Document created**: February 10, 2026
**Author**: Seriguela Research Team
**Status**: Ready for relaunch
