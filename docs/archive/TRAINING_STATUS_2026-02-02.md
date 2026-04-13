# Training Status - Model Scaling Experiment

**Date**: 2026-02-02 23:41:37
**Status**: 🟢 **TRAINING IN PROGRESS**

---

## Instances Running

| Model | Instance ID | IP Address | Status |
|-------|-------------|------------|--------|
| **Base (124M)** | i-0855711efcac25a9c | 18.206.190.126 | 🟢 Running |
| **Medium (355M)** | i-0eea77c3bbf1ea976 | 13.220.236.233 | 🟢 Running |
| **Large (774M)** | i-04dc6f51534d8185d | 52.55.119.255 | 🟢 Running |

**SSH Key**: chave-gpu-nova

---

## Monitoring

### 1. Check Instance Status

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-*-training" \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]" \
  --output table
```

### 2. Monitor Training Logs (SSH)

**Base Model**:
```bash
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@18.206.190.126
tail -f /home/ubuntu/training_base.log
```

**Medium Model**:
```bash
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@13.220.236.233
tail -f /home/ubuntu/training_medium.log
```

**Large Model**:
```bash
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@52.55.119.255
tail -f /home/ubuntu/training_large.log
```

### 3. Monitor Wandb Dashboard

Access: https://wandb.ai/YOUR_USERNAME/seriguela

Expected runs:
- seriguela-supervised-base-700k-TIMESTAMP
- seriguela-supervised-medium-700k-TIMESTAMP
- seriguela-supervised-large-700k-TIMESTAMP

### 4. Check Training Completion

**Check if training finished** (returns "DONE" when complete):
```bash
# Base
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@18.206.190.126 \
  'test -f ~/.training_complete && echo "DONE" || echo "Running"'

# Medium
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@13.220.236.233 \
  'test -f ~/.training_complete && echo "DONE" || echo "Running"'

# Large
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@52.55.119.255 \
  'test -f ~/.training_complete && echo "DONE" || echo "Running"'
```

---

## Expected Timeline

| Model | Start Time | Expected Duration | Expected Completion |
|-------|-----------|-------------------|---------------------|
| Base | 23:42:45 | 2-3 hours | ~01:42 - 02:42 |
| Medium | 23:43:00 | 3-4 hours | ~02:43 - 03:43 |
| Large | 23:43:16 | 4-5 hours | ~03:43 - 04:43 |

**Earliest completion**: ~01:42 (Base)
**Latest completion**: ~04:43 (Large)

---

## When Training Completes

### CRITICAL: Stop Instances Immediately

```bash
aws ec2 stop-instances --instance-ids i-0855711efcac25a9c i-0eea77c3bbf1ea976 i-04dc6f51534d8185d
```

**Or using filter**:
```bash
aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-*-training" "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].InstanceId" --output text)
```

### Verify All Stopped

```bash
aws ec2 describe-instances \
  --instance-ids i-0855711efcac25a9c i-0eea77c3bbf1ea976 i-04dc6f51534d8185d \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name]" \
  --output table
```

Expected output:
```
------------------------------------------
|         DescribeInstances             |
+----------------------+----------------+
|  i-0855711efcac25a9c |  stopped       |
|  i-0eea77c3bbf1ea976 |  stopped       |
|  i-04dc6f51534d8185d |  stopped       |
+----------------------+----------------+
```

---

## Download Trained Models

```bash
# Create output directory
mkdir -p output

# Base (124M)
scp -i C:\Users\madeinweb\chave-gpu.pem -r \
  ubuntu@18.206.190.126:~/seriguela/output/gpt2_base_700K_json \
  ./output/

# Medium (355M)
scp -i C:\Users\madeinweb\chave-gpu.pem -r \
  ubuntu@13.220.236.233:~/seriguela/output/gpt2_medium_700K_json \
  ./output/

# Large (774M)
scp -i C:\Users\madeinweb\chave-gpu.pem -r \
  ubuntu@52.55.119.255:~/seriguela/output/gpt2_large_700K_json \
  ./output/
```

---

## Cost Tracking

| Instance Type | Rate/hour | Expected Hours | Expected Cost |
|---------------|-----------|----------------|---------------|
| g5.xlarge (Base) | $1.006 | 2-3h | $2.01-3.02 |
| g5.xlarge (Medium) | $1.006 | 3-4h | $3.02-4.02 |
| g5.2xlarge (Large) | $1.212 | 4-5h | $4.85-6.06 |
| **Total** | - | **~10h** | **$9.88-13.10** |

**To calculate actual cost after training**:
1. Note exact termination times
2. Calculate duration for each instance
3. Multiply by hourly rates
4. Update `TRAINING_LOG_MODEL_SCALING_2025.md`

---

## Next Steps (After Training Completes)

1. ✅ Stop all instances
2. ✅ Verify all stopped
3. ✅ Download trained models
4. ✅ Update TRAINING_LOG with final times and costs
5. ⏳ Run evaluation: `bash scripts/run_nguyen_suite.sh`
6. ⏳ Aggregate results: `python scripts/aggregate_nguyen_results.py --input_dir nguyen_suite_results`
7. ⏳ Fill results in EXPERIMENT_MODEL_SCALING.md
8. ⏳ Update model cards with results
9. ⏳ Upload models to HuggingFace
10. ⏳ Git commit with all results

---

## Troubleshooting

### If training fails or stalls:

1. **Check logs via SSH**:
   ```bash
   ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@IP
   tail -100 /home/ubuntu/training_*.log
   ```

2. **Check system resources**:
   ```bash
   ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@IP
   nvidia-smi
   htop
   ```

3. **Check cloud-init status**:
   ```bash
   ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@IP
   tail -100 /var/log/user-data.log
   ```

4. **If need to restart**:
   - Stop failed instance
   - Terminate it (optional)
   - Re-run individual launch script:
     ```bash
     bash scripts/aws/launch_base_training.sh --wandb-key KEY --hf-token TOKEN
     ```

---

## Files Created

**Local tracking files**:
- `~/.seriguela/base_instance_info.txt`
- `~/.seriguela/medium_instance_info.txt`
- `~/.seriguela/large_instance_info.txt`

**Launch logs**:
- `aws_launch_logs/launch_base.log`
- `aws_launch_logs/launch_medium.log`
- `aws_launch_logs/launch_large.log`

**Documentation**:
- `TRAINING_LOG_MODEL_SCALING_2025.md` (updated with launch info)
- This status file: `TRAINING_STATUS_2026-02-02.md`

---

**Last Updated**: 2026-02-02 23:47:00

**Monitoring Recommended**: Check status every 30-60 minutes via Wandb or AWS console.
