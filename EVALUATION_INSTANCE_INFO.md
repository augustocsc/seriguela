# Evaluation Instance Information

## Instance Details
- **Instance ID**: i-051cad4bd51af8746
- **Type**: g5.2xlarge (NVIDIA A10G, 48GB VRAM)
- **Public IP**: 3.81.72.206
- **Launch Time**: 2026-02-11 04:03 UTC
- **Status**: Setting up (cloud-init in progress)

## Configuration
- **Models**: 4 (Base/Medium/Large prefix + Infix base)
- **Benchmarks**: Nguyen 1-12 (12 total)
- **Algorithms**: PPO + GRPO (2 total)
- **Epochs**: 20 per experiment
- **Total Experiments**: 4 models × 12 benchmarks × 2 algorithms = 96 experiments

## Estimated Timeline
- **Setup**: ~5-10 minutes (installing packages, downloading models)
- **Evaluation**: ~8-12 hours (96 experiments × 20 epochs each)
- **Total**: ~8-12 hours

## Access Commands

### SSH Access
```bash
ssh -i C:/Users/madeinweb/chave-gpu.pem ubuntu@3.81.72.206
```

### Monitor Setup Progress
```bash
ssh -i C:/Users/madeinweb/chave-gpu.pem ubuntu@3.81.72.206 'tail -f setup.log'
```

### Monitor Evaluation Progress
```bash
ssh -i C:/Users/madeinweb/chave-gpu.pem ubuntu@3.81.72.206 'tail -f seriguela/evaluation.log'
```

### Check GPU Status
```bash
ssh -i C:/Users/madeinweb/chave-gpu.pem ubuntu@3.81.72.206 'nvidia-smi'
```

### Download Results (when complete)
```bash
scp -r -i C:/Users/madeinweb/chave-gpu.pem ubuntu@3.81.72.206:~/seriguela/evaluation_results ./
```

## Cost Estimate
- **g5.2xlarge**: $1.212/hour
- **Estimated duration**: 12 hours
- **Estimated cost**: ~$14.54 USD

## IMPORTANT: Remember to Stop Instance!
```bash
aws ec2 stop-instances --instance-ids i-051cad4bd51af8746
```

## Monitoring Status

### 04:03 UTC - Instance Launched
- Instance started successfully
- Cloud-init setup in progress

### 04:05 UTC - Cloud-init Timeout
- Cloud-init wait timeout (300s) - this is normal for the user-data script
- System is proceeding with package installation

---
**Last Updated**: 2026-02-11 04:05 UTC
