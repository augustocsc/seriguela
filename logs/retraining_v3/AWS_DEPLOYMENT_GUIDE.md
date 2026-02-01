# AWS Deployment Guide - Model v3 Training

## Prerequisites

Before starting, ensure you have:

1. **AWS CLI configured** with valid credentials
   ```bash
   aws configure
   ```

2. **SSH Key available**: `C:/Users/madeinweb/chave-gpu.pem`

3. **HuggingFace Token** (for model hub push): Get from https://huggingface.co/settings/tokens

4. **Weights & Biases Key** (optional, for experiment tracking): Get from https://wandb.ai/authorize

5. **GitHub changes pushed** (completed ✓)

---

## Step-by-Step Deployment

### Step 1: Launch AWS EC2 Instance

**Option A: Using existing launch script**
```bash
bash scripts/aws/launch_evaluation_instance.sh --hf-token YOUR_HF_TOKEN
```

**Option B: Manual launch (if script not available)**
```bash
# Launch g5.xlarge instance with Ubuntu 22.04 Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g5.xlarge \
  --key-name seriguela-key \
  --security-group-ids sg-0deaa73e23482e3f6 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=seriguela-v3-training}]'
```

**Save instance details:**
```bash
# Get instance ID
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-v3-training" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].InstanceId" \
  --output text

# Get public IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-v3-training" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text
```

**Expected wait time:** 2-3 minutes for instance to start

---

### Step 2: Verify SSH Access

```bash
# Test SSH connection (replace PUBLIC_IP)
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP "echo 'SSH connection successful'"
```

**If connection fails:**
- Verify security group allows your IP:
  ```bash
  MY_IP=$(curl -s https://checkip.amazonaws.com)
  aws ec2 authorize-security-group-ingress \
    --group-id sg-0deaa73e23482e3f6 \
    --protocol tcp --port 22 --cidr $MY_IP/32
  ```

---

### Step 3: Setup Environment on AWS

```bash
# SSH into instance
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP
```

**On AWS instance:**

```bash
# Update system
sudo apt update

# Install required packages
sudo apt install -y python3-pip python3-venv git

# Clone repository
cd ~
git clone https://github.com/augustocsc/seriguela.git
cd seriguela

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
CUDA available: True
GPU 0: NVIDIA A10G (23GB)
```

---

### Step 4: Prepare Training Data

Since CSV files are not in git (too large), prepare them on AWS:

```bash
# Still on AWS instance, in ~/seriguela directory
cd ~/seriguela
source venv/bin/activate

# Run data preparation script
python scripts/data/prepare_training_data_fixed.py \
  --dataset_repo_id augustocsc/sintetico_natural \
  --data_dir 700K \
  --data_column i_prompt_n \
  --output_dir ./data/processed/700K_fixed \
  --validate
```

**Expected output:**
```
Total examples: 947876
Processed examples: 947876
Valid rate: 100.0%
✅ All examples validated successfully!
```

**Expected time:** 5-10 minutes

**Verify data files:**
```bash
ls -lh data/processed/700K_fixed/
# Should show:
# train_700K.csv (350MB)
# validation_700K.csv (45MB)
# test_700K.csv (44MB)
```

---

### Step 5: Configure Weights & Biases (Optional but Recommended)

```bash
# Login to wandb
wandb login

# Enter your API key when prompted
# Get key from: https://wandb.ai/authorize
```

**If you skip this step:** Training will proceed without experiment tracking (not recommended for production runs).

---

### Step 6: Configure HuggingFace Hub (Optional)

Only needed if you want to auto-push the trained model to HuggingFace Hub:

```bash
huggingface-cli login
# Enter your token when prompted
```

---

### Step 7: Start Training

**Run training in background with nohup:**

```bash
cd ~/seriguela
source venv/bin/activate

# Start training in background
nohup bash scripts/aws/train_v3_model.sh > train_v3_output.log 2>&1 &

# Get process ID
echo $! > train_v3.pid
```

**Expected training time:** 2-3 hours for 3 epochs on GPT-2 Small

**The script will automatically:**
1. Verify GPU availability
2. Check data files have end markers
3. Create output directory
4. Start training with wandb logging
5. Save checkpoints every epoch
6. Save final model

---

### Step 8: Monitor Training

**Option A: Watch log file (from local machine)**

```bash
# From your local Windows machine
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP "tail -f ~/seriguela/train_v3_output.log"
```

**Option B: Weights & Biases dashboard**

Visit: https://wandb.ai/YOUR_USERNAME/seriguela_v3

**Option C: SSH and check manually**

```bash
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP

# Check if training is running
ps aux | grep train.py

# Check GPU usage
watch -n 2 nvidia-smi

# Tail log
tail -50 ~/seriguela/train_v3_output.log
```

**What to look for in logs:**
- `Training...` messages with loss values decreasing
- `Epoch X/3` progress indicators
- No CUDA out-of-memory errors
- No "marker not found" errors

---

### Step 9: Wait for Training Completion

**Monitor completion:**

```bash
# From local machine - check if training process is still running
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP "pgrep -f train.py"

# Empty output = training finished
```

**Or use a monitoring loop:**

```bash
while ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP "pgrep -f train.py > /dev/null"; do
  echo "Training still running... ($(date))"
  sleep 300  # Check every 5 minutes
done
echo "Training complete!"
```

---

### Step 10: Run Evaluation

After training completes, evaluate the model:

```bash
# SSH to AWS instance
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP

cd ~/seriguela
source venv/bin/activate

# Run evaluation
python scripts/evaluate.py \
  --model_path ./output/Se124M_700K_infix_v3 \
  --base_model gpt2 \
  --num_samples 500 \
  --output_dir ./evaluation_results/v3
```

**Expected time:** 10-15 minutes

**Run generation test:**

```bash
python scripts/generate.py \
  --model_path ./output/Se124M_700K_infix_v3 \
  --base_model gpt2 \
  --num_generations 50 \
  --validate > generation_v3.log

# Check results
cat generation_v3.log | grep "Status:"
```

---

### Step 11: Download Results

From your local Windows machine:

```bash
# Download evaluation results
scp -i "C:/Users/madeinweb/chave-gpu.pem" \
  ubuntu@PUBLIC_IP:~/seriguela/evaluation_results/v3/*.json \
  ./logs/retraining_v3/

# Download generation log
scp -i "C:/Users/madeinweb/chave-gpu.pem" \
  ubuntu@PUBLIC_IP:~/seriguela/generation_v3.log \
  ./logs/retraining_v3/

# Download training log
scp -i "C:/Users/madeinweb/chave-gpu.pem" \
  ubuntu@PUBLIC_IP:~/seriguela/train_v3_output.log \
  ./logs/retraining_v3/
```

---

### Step 12: Analyze Results Locally

```bash
cd logs/retraining_v3

# Check metrics
cat metrics_*.json | grep -E "valid_rate|parseable_rate"

# Check sample generations
tail -50 generation_v3.log | grep "Status: VALID"

# Check for issues
grep -i "error\|warning\|failed" train_v3_output.log
```

**Success criteria:**
- ✅ Valid rate > 80%
- ✅ Expressions stop at `<|endofex|>` (no concatenation)
- ✅ No garbage tokens like "BuyableInstoreAndOnline"
- ✅ Uses only valid variables (x_1, x_2, etc.) and operators

---

### Step 13: Push Model to HuggingFace Hub (If Successful)

If evaluation results are good:

```bash
# SSH to AWS
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP

cd ~/seriguela
source venv/bin/activate

# Push to Hub (if you configured HuggingFace CLI earlier)
huggingface-cli upload \
  augustocsc/Se124M_700K_infix_v3 \
  ./output/Se124M_700K_infix_v3
```

**Alternative: Download model and push locally**

```bash
# Download model (large transfer, ~500MB)
scp -r -i "C:/Users/madeinweb/chave-gpu.pem" \
  ubuntu@PUBLIC_IP:~/seriguela/output/Se124M_700K_infix_v3 \
  ./output/

# Push from local machine
huggingface-cli upload augustocsc/Se124M_700K_infix_v3 ./output/Se124M_700K_infix_v3
```

---

### Step 14: Stop AWS Instance

**IMPORTANT:** Always stop the instance when done to avoid charges!

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-v3-training" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].InstanceId" \
  --output text)

# Stop instance
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Verify stopped
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query "Reservations[0].Instances[0].State.Name" \
  --output text
```

**Expected output:** `stopping` or `stopped`

---

### Step 15: Update Documentation

1. Update `logs/retraining_v3/RETRAINING_LOG.md` with:
   - Final metrics
   - Sample outputs
   - Issues encountered
   - Deployment decision

2. Create summary report:

```bash
cat > logs/retraining_v3/SUMMARY.md << EOF
# Model v3 Training Summary

## Problem
- v1: Generated valid expressions but didn't stop
- v2: Generated garbage due to missing end markers

## Solution
- Prepared 947K examples with proper <|endofex|> markers
- 100% validation rate before training

## Results
- Valid rate: XX%
- Stopping behavior: [FIXED/PARTIAL/BROKEN]
- Sample outputs: [paste examples]

## Recommendation
[DEPLOY / NEEDS_MORE_WORK / REJECT]

## Trained Model
- HuggingFace Hub: augustocsc/Se124M_700K_infix_v3
- Training date: $(date +%Y-%m-%d)
- Training time: XX hours
EOF
```

3. Commit results:

```bash
git add logs/retraining_v3/
git commit -m "Complete v3 retraining - results documented"
git push origin main
```

---

## Troubleshooting

### Issue: SSH connection refused

**Solution:**
```bash
# Add your IP to security group
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id sg-0deaa73e23482e3f6 \
  --protocol tcp --port 22 --cidr $MY_IP/32
```

### Issue: CUDA out of memory

**Solution:**
- Reduce batch size in config: `"per_device_train_batch_size": 4`
- Increase gradient accumulation: `"gradient_accumulation_steps": 8`

### Issue: Data preparation fails

**Solution:**
- Check HuggingFace Hub is accessible
- Verify dataset exists: https://huggingface.co/datasets/augustocsc/sintetico_natural
- Try downloading manually and using local files

### Issue: Training hangs

**Solution:**
- Check GPU usage: `nvidia-smi`
- Check logs: `tail -100 train_v3_output.log`
- Verify data files exist: `ls -lh data/processed/700K_fixed/`

### Issue: Model generates garbage

**Solution:**
- Verify end markers in data: `grep -c "<|endofex|>" data/processed/700K_fixed/train_700K.csv`
- Check training didn't error: `grep -i error train_v3_output.log`
- May need more epochs or different learning rate

---

## Cost Estimation

**AWS g5.xlarge pricing:** ~$1.00/hour (us-east-1)

**Expected costs:**
- Instance setup: 0.1 hours = $0.10
- Data preparation: 0.2 hours = $0.20
- Training (3 epochs): 2.5 hours = $2.50
- Evaluation: 0.3 hours = $0.30
- **Total:** ~$3.10 per training run

**Cost saving tips:**
- Stop instance immediately after downloading results
- Use spot instances for 70% discount (but risk interruption)
- Reuse same instance for multiple experiments

---

## Quick Reference Commands

```bash
# Launch instance
bash scripts/aws/launch_evaluation_instance.sh --hf-token TOKEN

# SSH
ssh -i "C:/Users/madeinweb/chave-gpu.pem" ubuntu@PUBLIC_IP

# Check training status
ssh -i "KEY" ubuntu@IP "pgrep -f train.py"

# Monitor logs
ssh -i "KEY" ubuntu@IP "tail -f ~/seriguela/train_v3_output.log"

# Download results
scp -i "KEY" ubuntu@IP:~/seriguela/evaluation_results/v3/*.json ./logs/retraining_v3/

# Stop instance
aws ec2 stop-instances --instance-ids INSTANCE_ID
```

---

## Next Steps After Successful v3

1. **If v3 succeeds (>80% valid rate):**
   - Deploy as production model
   - Update README with v3 examples
   - Consider training GPT-2 Medium (355M) for better quality

2. **If v3 partially succeeds (40-80% valid rate):**
   - Investigate failure patterns
   - Augment training data
   - Adjust hyperparameters

3. **If v3 fails (<40% valid rate):**
   - Verify data preparation worked
   - Review training logs for anomalies
   - Consider alternative approaches (T5, BART, curriculum learning)
