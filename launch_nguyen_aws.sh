#!/bin/bash
# Launch AWS instance for Nguyen benchmarks evaluation
# Runs 3 models × 12 benchmarks = 36 experiments with R² scoring

set -e

echo "=========================================="
echo "LAUNCHING AWS INSTANCE - NGUYEN BENCHMARKS"
echo "=========================================="

# Configuration
INSTANCE_TYPE="g5.xlarge"  # NVIDIA A10G, 24GB VRAM
KEY_NAME="chave-gpu-nova"
SECURITY_GROUP="sg-0deaa73e23482e3f6"
REGION="us-east-1"

# Find latest Deep Learning AMI
echo "Finding latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' \
  --output text \
  --region $REGION)

echo "Using AMI: $AMI_ID"

# Launch instance
echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=seriguela-nguyen-benchmarks}]' \
  --user-data file://aws/temp/userdata_nguyen_benchmarks.sh \
  --region $REGION \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $INSTANCE_ID"

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $REGION)

echo ""
echo "=========================================="
echo "INSTANCE READY"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Instance Type: $INSTANCE_TYPE"
echo ""
echo "Waiting 60 seconds for cloud-init..."
sleep 60

# Upload project files
echo "Uploading project files..."
KEY_PATH="C:/Users/madeinweb/chave-gpu.pem"

# Create directories
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP 'mkdir -p ~/seriguela/{scripts,data/benchmarks/nguyen,output,classes}'

# Upload scripts
echo "Uploading scripts..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
  scripts/evaluate_nguyen_benchmarks.py \
  scripts/run_all_nguyen_benchmarks.py \
  ubuntu@$PUBLIC_IP:~/seriguela/scripts/

# Upload classes
echo "Uploading expression class..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
  classes/expression.py \
  ubuntu@$PUBLIC_IP:~/seriguela/classes/

# Upload benchmarks
echo "Uploading Nguyen benchmarks..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
  data/benchmarks/nguyen/*.csv \
  data/benchmarks/nguyen/*.txt \
  ubuntu@$PUBLIC_IP:~/seriguela/data/benchmarks/nguyen/

# Compress and upload models
echo "Compressing models..."
if [ ! -f models_compressed.tar.gz ]; then
  tar -czf models_compressed.tar.gz \
    output/gpt2_base_700K_json \
    output/gpt2_medium_700K_json \
    output/gpt2_large_700K_json
fi

echo "Uploading models (155MB compressed)..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
  models_compressed.tar.gz \
  ubuntu@$PUBLIC_IP:~/seriguela/

# Extract models
echo "Extracting models on remote..."
ssh -i "$KEY_PATH" ubuntu@$PUBLIC_IP 'cd ~/seriguela && tar -xzf models_compressed.tar.gz && rm models_compressed.tar.gz'

# Start execution
echo ""
echo "=========================================="
echo "STARTING NGUYEN BENCHMARK EVALUATION"
echo "=========================================="
echo "Running 36 experiments (3 models × 12 benchmarks × 100 samples)"
echo "Estimated time: 2-3 hours"
echo ""

ssh -i "$KEY_PATH" ubuntu@$PUBLIC_IP << 'ENDSSH'
cd ~/seriguela
nohup python3 scripts/run_all_nguyen_benchmarks.py \
  --models base medium large \
  --benchmarks 1 2 3 4 5 6 7 8 9 10 11 12 \
  --num_samples 100 \
  --output_dir ./results_nguyen_benchmarks \
  --models_dir ./output \
  > nguyen_benchmarks.log 2>&1 &
echo $! > nguyen_benchmark.pid
ENDSSH

echo ""
echo "=========================================="
echo "EXECUTION STARTED"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  ssh -i \"$KEY_PATH\" ubuntu@$PUBLIC_IP"
echo "  tail -f ~/seriguela/nguyen_benchmarks.log"
echo ""
echo "Check completion:"
echo "  ssh -i \"$KEY_PATH\" ubuntu@$PUBLIC_IP 'grep \"SUITE COMPLETE\" ~/seriguela/nguyen_benchmarks.log'"
echo ""
echo "Download results when complete:"
echo "  scp -i \"$KEY_PATH\" -r ubuntu@$PUBLIC_IP:~/seriguela/results_nguyen_benchmarks ./"
echo ""
echo "STOP INSTANCE when done:"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""

# Save instance info
cat > nguyen_instance_info.txt << EOF
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Instance Type: $INSTANCE_TYPE
Key: $KEY_PATH
Launch Time: $(date)

Monitor:
ssh -i "$KEY_PATH" ubuntu@$PUBLIC_IP 'tail -f ~/seriguela/nguyen_benchmarks.log'

Download:
scp -i "$KEY_PATH" -r ubuntu@$PUBLIC_IP:~/seriguela/results_nguyen_benchmarks ./

Stop:
aws ec2 stop-instances --instance-ids $INSTANCE_ID
EOF

echo "Instance info saved to: nguyen_instance_info.txt"
echo ""
echo "🚀 Nguyen benchmarks evaluation STARTED!"
echo "⏱️  Estimated completion: $(date -d '+3 hours' '+%H:%M' 2>/dev/null || date -v +3H '+%H:%M' 2>/dev/null || echo 'in ~3 hours')"
