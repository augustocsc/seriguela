# Experiment Training Instances

**Date:** 2026-02-01
**Status:** COMPLETE - Instances Stopped

---

## Instance Details

| Experiment | Instance ID | Public IP | Status |
|------------|-------------|-----------|--------|
| EXP-A (JSON) | i-072bdda1680d59e20 | 54.166.216.158 | Running |
| EXP-B (EOS) | i-0b2a5321afd452f78 | 3.84.144.68 | Running |

---

## SSH Commands

### EXP-A (JSON Format)
```bash
# Connect
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.166.216.158

# Check setup progress
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.166.216.158 'tail -50 /var/log/user-data.log'

# Check training progress
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.166.216.158 'tail -50 /home/ubuntu/training_exp_a.log'

# Check if complete
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.166.216.158 'ls -la /home/ubuntu/.exp_a_complete 2>/dev/null && echo "COMPLETE" || echo "IN PROGRESS"'
```

### EXP-B (EOS Format)
```bash
# Connect
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@3.84.144.68

# Check setup progress
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@3.84.144.68 'tail -50 /var/log/user-data.log'

# Check training progress
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@3.84.144.68 'tail -50 /home/ubuntu/training_exp_b.log'

# Check if complete
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@3.84.144.68 'ls -la /home/ubuntu/.exp_b_complete 2>/dev/null && echo "COMPLETE" || echo "IN PROGRESS"'
```

---

## Download Results

### After Training Completes:
```bash
# EXP-A Results
scp -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@54.166.216.158:/home/ubuntu/seriguela/output/exp_a_json/evaluation_results.json ./results/

# EXP-B Results
scp -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@3.84.144.68:/home/ubuntu/seriguela/output/exp_b_eos/evaluation_results.json ./results/
```

---

## Stop Instances

```bash
aws ec2 stop-instances --instance-ids i-072bdda1680d59e20 i-0b2a5321afd452f78
```

---

## Expected Timeline

- Setup: ~10-15 minutes
- Data preparation: ~10-15 minutes
- Training (3 epochs): ~2-3 hours
- Evaluation: ~15 minutes

**Total:** ~3-4 hours per instance

---

## Cost Estimate

- g5.xlarge: ~$1.00/hour
- 2 instances x 4 hours = ~$8.00 total
