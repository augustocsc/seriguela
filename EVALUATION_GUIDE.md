# Model Evaluation Guide

## Overview

This guide walks you through evaluating the two Seriguela models to determine if adding the `<|endofex|>` ending token improved the model's ability to stop generation properly.

**Models to Compare:**
- **Model 1 (Original)**: `augustocsc/Se124M_700K_infix` - Trained without ending token
- **Model 2 (V2)**: `augustocsc/Se124M_700K_infix_v2` - Trained with `<|endofex|>` ending token

## Quick Start

### 1. Launch AWS Evaluation Instance

From your local machine (Windows):

```bash
# Make the script executable (if on Git Bash/WSL)
chmod +x scripts/aws/launch_evaluation_instance.sh

# Launch instance (HF token optional for public models)
bash scripts/aws/launch_evaluation_instance.sh --hf-token YOUR_HF_TOKEN
```

**Note:** The script will automatically:
- Launch a g5.xlarge instance with GPU
- Install all dependencies
- Clone the repository
- Set up the environment
- Take approximately 5-10 minutes to complete setup

### 2. Wait for Setup to Complete

The script will output connection commands. To monitor setup:

```bash
# Replace PUBLIC_IP with the IP shown by the launch script
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@PUBLIC_IP 'tail -f /var/log/user-data.log'
```

Or wait for completion:

```bash
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@PUBLIC_IP 'while [ ! -f ~/.setup_complete ]; do sleep 10; echo "Setup in progress..."; done; echo "✅ Setup complete!"; cat ~/setup_info.txt'
```

### 3. Run Evaluation

Once setup is complete, SSH into the instance and run the evaluation:

```bash
# Connect to instance
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@PUBLIC_IP

# Activate environment and run evaluation
cd seriguela
source venv/bin/activate
bash scripts/aws/evaluate_models.sh
```

**Or run in background:**

```bash
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@PUBLIC_IP 'cd seriguela && source venv/bin/activate && nohup bash scripts/aws/evaluate_models.sh > evaluation.log 2>&1 &'
```

### 4. Monitor Progress

From your local machine:

```bash
bash scripts/aws/monitor_evaluation.sh PUBLIC_IP
```

This will:
- Show real-time evaluation progress
- Automatically download results when complete
- Display comparison metrics

### 5. Review Results

Results will be saved in `./evaluation_results/comparison/`:

```bash
# View comparison summary
cat ./evaluation_results/comparison/comparison_*.json | jq '.comparison'

# View full results
cat ./evaluation_results/comparison/comparison_*.json | jq
```

### 6. Stop Instance When Done

**IMPORTANT:** Don't forget to stop the instance to avoid charges!

```bash
# Get instance ID from saved info
INSTANCE_ID=$(cat ~/.seriguela/last_evaluation_instance_id.txt)

# Stop instance
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Or terminate if you don't need it anymore
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

## What Gets Evaluated

The evaluation script compares both models on **500 test samples** and measures:

### Key Metrics

1. **Valid Rate**: Percentage of syntactically correct expressions
2. **Parseable Rate**: Percentage of expressions that can be parsed
3. **Constraints Met**: Percentage following prompt constraints (variables, operators)
4. **Diversity**: Unique expressions generated
5. **Expression Length**: Average length of generated expressions

### Expected Improvements with V2

If the `<|endofex|>` token helped, you should see:
- ✅ **Higher valid rate** - Model generates more correct expressions
- ✅ **Better stopping** - Expressions end cleanly without concatenation
- ✅ **Shorter expressions** - No extra text after the expression
- ✅ **Cleaner boundaries** - Clear start and end of expressions

## Manual Evaluation (Alternative)

If you prefer to run evaluations manually:

### Evaluate Single Model

```bash
python scripts/evaluate.py \
    --model_path augustocsc/Se124M_700K_infix_v2 \
    --num_samples 500 \
    --output_dir ./evaluation_results/v2
```

### Compare Two Models

```bash
python scripts/compare_models.py \
    --model1 augustocsc/Se124M_700K_infix \
    --model2 augustocsc/Se124M_700K_infix_v2 \
    --model1_name "Original" \
    --model2_name "With End Token" \
    --num_samples 500 \
    --output_dir ./evaluation_results/comparison
```

## Understanding Results

### Comparison Output Example

```json
{
  "comparison": {
    "valid_rate_diff": 0.23,        // +23% improvement
    "parseable_rate_diff": 0.19,    // +19% improvement
    "constraints_met_diff": 0.15,   // +15% improvement
    "diversity_diff": 0.05          // +5% improvement
  }
}
```

### Interpretation

- **Positive values** = V2 (with end token) is better
- **Negative values** = Original model is better
- **Near zero** = No significant difference

### Success Criteria

The `<|endofex|>` token is successful if:
- `valid_rate_diff > 0.10` (10%+ improvement)
- Expressions end cleanly (check sample outputs)
- No concatenation issues in generated expressions

## Troubleshooting

### Setup Fails

Check the setup log:
```bash
ssh ubuntu@PUBLIC_IP 'tail -100 /var/log/user-data.log'
```

### GPU Not Available

Verify GPU:
```bash
ssh ubuntu@PUBLIC_IP 'nvidia-smi'
```

If no GPU, evaluation will still work but be slower.

### Model Download Issues

If models fail to download, check:
1. HuggingFace token is valid (for gated models)
2. Internet connectivity: `ssh ubuntu@PUBLIC_IP 'ping -c 3 huggingface.co'`
3. Disk space: `ssh ubuntu@PUBLIC_IP 'df -h'`

### Out of Memory

If evaluation runs out of memory:
- Reduce `--num_samples` in the evaluation script
- Use a larger instance type: `--instance-type g5.2xlarge`

## Cost Estimation

- **g5.xlarge**: ~$1.00/hour
- **Evaluation time**: ~30-60 minutes for 500 samples
- **Total cost**: ~$0.50-$1.00 per evaluation run

**Tip:** Always stop instances when done!

## Next Steps

After evaluation:

1. **Review comparison results** to see if V2 improved
2. **Examine sample outputs** to understand generation quality
3. **Check error distribution** to identify remaining issues
4. **Decide next steps**:
   - If V2 is better: Deploy and use V2
   - If similar: Investigate other improvements
   - If worse: Debug training setup

## Files Created

- `scripts/aws/launch_evaluation_instance.sh` - Launch AWS instance
- `scripts/aws/evaluate_models.sh` - Run evaluation on instance
- `scripts/aws/monitor_evaluation.sh` - Monitor and download results
- `evaluation_results/comparison/` - Comparison results and metrics

## Quick Reference Commands

```bash
# Launch instance
bash scripts/aws/launch_evaluation_instance.sh --hf-token YOUR_TOKEN

# Monitor evaluation
bash scripts/aws/monitor_evaluation.sh PUBLIC_IP

# Stop instance
aws ec2 stop-instances --instance-ids $(cat ~/.seriguela/last_evaluation_instance_id.txt)

# View results
cat ./evaluation_results/comparison/comparison_*.json | jq
```
