#!/bin/bash
# Complete evaluation workflow on AWS
# Created: 2026-02-10
# 1. Download Base and Medium models
# 2. Launch evaluation instance
# 3. Upload models
# 4. Run evaluations
# 5. Download results

set -e

SSH_KEY="C:/Users/madeinweb/chave-gpu.pem"
BASE_ID="i-03cb806bdc98e6d36"
MEDIUM_ID="i-0567ed93f9e625a89"
LOCAL_OUTPUT="./output"
RESULTS_DIR="./evaluation_results_aws"

echo "============================================"
echo "AWS Evaluation Workflow"
echo "============================================"
echo ""

# ============================================
# STEP 1: Download Base and Medium models
# ============================================
echo "STEP 1: Downloading Base and Medium models..."
echo "============================================"
echo ""

# Check if instances are stopped
echo "Checking instance states..."
BASE_STATE=$(aws ec2 describe-instances --instance-ids $BASE_ID --query 'Reservations[0].Instances[0].State.Name' --output text)
MEDIUM_STATE=$(aws ec2 describe-instances --instance-ids $MEDIUM_ID --query 'Reservations[0].Instances[0].State.Name' --output text)

echo "Base state: $BASE_STATE"
echo "Medium state: $MEDIUM_STATE"
echo ""

# Start instances if stopped
if [ "$BASE_STATE" = "stopped" ] || [ "$MEDIUM_STATE" = "stopped" ]; then
  echo "Starting instances temporarily..."
  aws ec2 start-instances --instance-ids $BASE_ID $MEDIUM_ID

  echo "Waiting for instances to be running..."
  aws ec2 wait instance-running --instance-ids $BASE_ID $MEDIUM_ID

  echo "Waiting additional 30s for SSH to be ready..."
  sleep 30
fi

# Get IP addresses
BASE_IP=$(aws ec2 describe-instances --instance-ids $BASE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
MEDIUM_IP=$(aws ec2 describe-instances --instance-ids $MEDIUM_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Base IP: $BASE_IP"
echo "Medium IP: $MEDIUM_IP"
echo ""

# Download models if not already local
mkdir -p $LOCAL_OUTPUT

if [ ! -d "$LOCAL_OUTPUT/gpt2_base_prefix_682k" ]; then
  echo "Downloading Base model..."
  scp -i $SSH_KEY -o StrictHostKeyChecking=no -r \
    ubuntu@$BASE_IP:~/seriguela/output/gpt2_base_prefix_682k \
    $LOCAL_OUTPUT/ || echo "Warning: Could not download Base model"
  echo "✓ Base downloaded"
else
  echo "✓ Base model already exists locally"
fi

if [ ! -d "$LOCAL_OUTPUT/gpt2_medium_prefix_682k" ]; then
  echo "Downloading Medium model..."
  scp -i $SSH_KEY -o StrictHostKeyChecking=no -r \
    ubuntu@$MEDIUM_IP:~/seriguela/output/gpt2_medium_prefix_682k \
    $LOCAL_OUTPUT/ || echo "Warning: Could not download Medium model"
  echo "✓ Medium downloaded"
else
  echo "✓ Medium model already exists locally"
fi

echo ""

# Stop instances to save costs
echo "Stopping Base and Medium instances..."
aws ec2 stop-instances --instance-ids $BASE_ID $MEDIUM_ID
echo "✓ Instances stopped"
echo ""

# Verify models downloaded
if [ ! -d "$LOCAL_OUTPUT/gpt2_base_prefix_682k" ]; then
  echo "ERROR: Base model not found locally!"
  exit 1
fi

if [ ! -d "$LOCAL_OUTPUT/gpt2_medium_prefix_682k" ]; then
  echo "ERROR: Medium model not found locally!"
  exit 1
fi

echo "✓ Both models ready for upload"
echo ""

# ============================================
# STEP 2: Launch evaluation instance
# ============================================
echo "STEP 2: Launching evaluation instance..."
echo "============================================"
echo ""

INSTANCE_TYPE="g5.xlarge"
IMAGE_ID="ami-01dfe92df9055a1c6"
KEY_NAME="chave-gpu-nova"
SECURITY_GROUP="sg-0deaa73e23482e3f6"
VOLUME_SIZE=80

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_NAME="seriguela-evaluation-${TIMESTAMP}"

# Get credentials
WANDB_KEY=$(cat ~/.tokens.txt 2>/dev/null | grep wandb | cut -d= -f2 | tr -d ' ' || echo "")
HF_TOKEN=$(cat ~/.tokens.txt 2>/dev/null | grep huggingface | cut -d= -f2 | tr -d ' ' || echo "")

if [ -z "$WANDB_KEY" ] || [ -z "$HF_TOKEN" ]; then
  echo "ERROR: Could not read credentials from ~/.tokens.txt"
  echo "Please ensure file exists with format:"
  echo "  huggingface = hf_..."
  echo "  wandb = ..."
  exit 1
fi

# Create user data
TEMP_DIR="C:/Users/madeinweb/AppData/Local/Temp"
cat > "$TEMP_DIR/userdata_eval.sh" <<'EOF'
#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log) 2>&1

sleep 5

cd /home/ubuntu
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
git checkout experiment/ppo-symbolic-regression

PYTHON=/opt/pytorch/bin/python3
PIP=/opt/pytorch/bin/pip3

$PIP install -r requirements.txt

echo "huggingface = HFTOKEN_PLACEHOLDER" > /home/ubuntu/.tokens.txt
echo "wandb = WBKEY_PLACEHOLDER" >> /home/ubuntu/.tokens.txt
chmod 600 /home/ubuntu/.tokens.txt

export HF_TOKEN="HFTOKEN_PLACEHOLDER"
export WANDB_API_KEY="WBKEY_PLACEHOLDER"
/opt/pytorch/bin/huggingface-cli login --token $HF_TOKEN
/opt/pytorch/bin/wandb login $WANDB_API_KEY

mkdir -p output

touch /home/ubuntu/.eval_ready
echo "Ready for model upload at $(date)" >> /home/ubuntu/.eval_ready
EOF

# Replace tokens
sed -i "s|HFTOKEN_PLACEHOLDER|${HF_TOKEN}|g" "$TEMP_DIR/userdata_eval.sh"
sed -i "s|WBKEY_PLACEHOLDER|${WANDB_KEY}|g" "$TEMP_DIR/userdata_eval.sh"

# Launch instance
echo "Launching instance $INSTANCE_NAME..."
EVAL_INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $IMAGE_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\"}}]" \
  --user-data file://$TEMP_DIR/userdata_eval.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $EVAL_INSTANCE_ID"
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $EVAL_INSTANCE_ID

EVAL_IP=$(aws ec2 describe-instances \
  --instance-ids $EVAL_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance ready at: $EVAL_IP"
echo "Waiting 60s for setup to complete..."
sleep 60

rm "$TEMP_DIR/userdata_eval.sh"
echo ""

# ============================================
# STEP 3: Upload models to evaluation instance
# ============================================
echo "STEP 3: Uploading models to evaluation instance..."
echo "============================================"
echo ""

echo "Uploading Base model..."
scp -i $SSH_KEY -o StrictHostKeyChecking=no -r \
  $LOCAL_OUTPUT/gpt2_base_prefix_682k \
  ubuntu@$EVAL_IP:~/seriguela/output/

echo "Uploading Medium model..."
scp -i $SSH_KEY -o StrictHostKeyChecking=no -r \
  $LOCAL_OUTPUT/gpt2_medium_prefix_682k \
  ubuntu@$EVAL_IP:~/seriguela/output/

echo "✓ Models uploaded"
echo ""

# ============================================
# STEP 4: Run evaluations
# ============================================
echo "STEP 4: Running evaluations on AWS..."
echo "============================================"
echo ""

# Create evaluation script on remote instance
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$EVAL_IP <<'REMOTE_EVAL'
cd ~/seriguela

PYTHON=/opt/pytorch/bin/python3
RESULTS_DIR="evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "============================================"
echo "Running Evaluations"
echo "============================================"

# Quick validation
echo "1. Quick validation (5 samples each)..."
for model in base medium; do
  $PYTHON scripts/generate.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_generations 5 \
    --validate \
    > $RESULTS_DIR/${model}_quick_samples.txt 2>&1 || true
done

# Quality metrics
echo "2. Quality metrics (500 samples each)..."
for model in base medium; do
  $PYTHON scripts/evaluate.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_samples 500 \
    --output_file $RESULTS_DIR/${model}_quality_metrics.json \
    2>&1 | tee $RESULTS_DIR/${model}_quality.log || true
done

# Complexity analysis
echo "3. Complexity analysis (200 samples each)..."
for model in base medium; do
  $PYTHON scripts/analyze_complexity.py \
    --model_path ./output/gpt2_${model}_prefix_682k \
    --num_samples 200 \
    --output_file $RESULTS_DIR/complexity_${model}_prefix.json \
    2>&1 | tee $RESULTS_DIR/${model}_complexity.log || true
done

# Model comparison
echo "4. Model comparison (Nguyen-5)..."
if [ -f "data/benchmarks/nguyen/nguyen_5.csv" ]; then
  $PYTHON scripts/compare_trained_models.py \
    --model_base ./output/gpt2_base_prefix_682k \
    --model_medium ./output/gpt2_medium_prefix_682k \
    --dataset data/benchmarks/nguyen/nguyen_5.csv \
    --epochs 10 \
    --output_file $RESULTS_DIR/comparison_base_medium_nguyen5.json \
    2>&1 | tee $RESULTS_DIR/comparison.log || true
fi

# Compare with infix models if available
echo "5. Prefix vs Infix comparison..."
if [ -d "./output/gpt2_base_700K_json" ]; then
  $PYTHON scripts/compare_models.py \
    --model1 ./output/gpt2_base_prefix_682k \
    --model2 ./output/gpt2_base_700K_json \
    --num_samples 500 \
    --output_file $RESULTS_DIR/prefix_vs_infix_base.json \
    2>&1 | tee $RESULTS_DIR/prefix_vs_infix.log || true
fi

# Generate summary
cat > $RESULTS_DIR/SUMMARY.md <<SUMMARY_EOF
# Evaluation Results: Base vs Medium (Prefix Notation)

**Date**: $(date)
**Models Evaluated**: Base (124M), Medium (355M)

## Files Generated

$(ls -lh $RESULTS_DIR/*.json $RESULTS_DIR/*.txt 2>/dev/null | awk '{print "- " $9 " (" $5 ")"}')

## Evaluation Complete

All results saved to: $RESULTS_DIR/

SUMMARY_EOF

echo ""
echo "============================================"
echo "Evaluations Complete!"
echo "============================================"
echo "Results saved to: $RESULTS_DIR/"

# Mark completion
touch ~/.evaluation_complete
echo "Evaluation completed at $(date)" >> ~/.evaluation_complete

# Show summary
cat $RESULTS_DIR/SUMMARY.md
REMOTE_EVAL

echo ""
echo "✓ Evaluations complete"
echo ""

# ============================================
# STEP 5: Download results
# ============================================
echo "STEP 5: Downloading results..."
echo "============================================"
echo ""

mkdir -p $RESULTS_DIR

# Get results directory name from remote
REMOTE_RESULTS_DIR=$(ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$EVAL_IP \
  'ls -td ~/seriguela/evaluation_results_* 2>/dev/null | head -1')

echo "Downloading from: $REMOTE_RESULTS_DIR"

scp -i $SSH_KEY -o StrictHostKeyChecking=no -r \
  ubuntu@$EVAL_IP:$REMOTE_RESULTS_DIR \
  $RESULTS_DIR/

echo "✓ Results downloaded to: $RESULTS_DIR/"
echo ""

# ============================================
# STEP 6: Stop evaluation instance
# ============================================
echo "STEP 6: Stopping evaluation instance..."
echo "============================================"
echo ""

read -p "Stop evaluation instance $EVAL_INSTANCE_ID? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  aws ec2 stop-instances --instance-ids $EVAL_INSTANCE_ID
  echo "✓ Instance stopped"
else
  echo "⚠️  Instance still running: $EVAL_INSTANCE_ID ($EVAL_IP)"
  echo "   Stop it manually when done: aws ec2 stop-instances --instance-ids $EVAL_INSTANCE_ID"
fi

echo ""

# ============================================
# COMPLETION SUMMARY
# ============================================
echo "============================================"
echo "EVALUATION WORKFLOW COMPLETE!"
echo "============================================"
echo ""
echo "Evaluation instance: $EVAL_INSTANCE_ID ($EVAL_IP)"
echo "Results location: $RESULTS_DIR/"
echo ""
echo "To view results:"
echo "  cd $RESULTS_DIR"
echo "  cat */SUMMARY.md"
echo "  cat */*.json | jq ."
echo ""
echo "Cost estimate:"
echo "  Instance runtime: ~1-2 hours"
echo "  Cost: ~$1-2 USD (g5.xlarge)"
echo ""

# Save instance info
cat > evaluation_aws_info.txt <<EOF
EVALUATION_INSTANCE_ID=$EVAL_INSTANCE_ID
EVALUATION_IP=$EVAL_IP
RESULTS_DIR=$RESULTS_DIR
LAUNCHED_AT=$(date)
EOF

echo "Instance info saved to: evaluation_aws_info.txt"
echo ""
