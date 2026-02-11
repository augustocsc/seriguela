#!/bin/bash
# Download trained models from AWS and stop instances
# Created: 2026-02-10

set -e

SSH_KEY="C:/Users/madeinweb/chave-gpu.pem"
OUTPUT_DIR="./output"

echo "============================================"
echo "Download Models & Stop AWS Instances"
echo "============================================"
echo ""

# Instance information (from launch logs)
BASE_IP="3.233.238.126"
BASE_ID="i-03cb806bdc98e6d36"
MEDIUM_IP="100.52.210.14"
MEDIUM_ID="i-0567ed93f9e625a89"
LARGE_IP="18.206.201.220"
LARGE_ID="i-060e3e00d1138c964"

# ============================================
# Function: Check training completion
# ============================================
check_completion() {
  local NAME=$1
  local IP=$2

  echo "Checking $NAME training status..."
  STATUS=$(ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$IP \
    'test -f ~/.training_complete && echo "DONE" || echo "Running"' 2>/dev/null || echo "ERROR")

  echo "  $NAME: $STATUS"
  echo "$STATUS"
}

# ============================================
# Function: Download model
# ============================================
download_model() {
  local NAME=$1
  local IP=$2
  local MODEL_NAME=$3

  echo ""
  echo "Downloading $NAME model from $IP..."
  scp -i $SSH_KEY -o StrictHostKeyChecking=no -r \
    ubuntu@$IP:~/seriguela/output/$MODEL_NAME \
    $OUTPUT_DIR/

  if [ $? -eq 0 ]; then
    echo "✓ $NAME model downloaded successfully"
    return 0
  else
    echo "✗ Failed to download $NAME model"
    return 1
  fi
}

# ============================================
# Function: Get training log tail
# ============================================
show_log() {
  local NAME=$1
  local IP=$2
  local LOG_FILE=$3

  echo ""
  echo "--- Last 20 lines of $NAME training log ---"
  ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$IP \
    "tail -20 ~/seriguela/$LOG_FILE" 2>/dev/null || echo "Could not fetch log"
  echo ""
}

# ============================================
# Check status of all instances
# ============================================
echo "Step 1: Checking instance status..."
aws ec2 describe-instances \
  --instance-ids $BASE_ID $MEDIUM_ID $LARGE_ID \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name,Tags[?Key=='Name'].Value|[0]]" \
  --output table

echo ""

# ============================================
# Check training completion
# ============================================
echo "Step 2: Checking training completion..."
BASE_STATUS=$(check_completion "BASE" $BASE_IP)
MEDIUM_STATUS=$(check_completion "MEDIUM" $MEDIUM_IP)
LARGE_STATUS=$(check_completion "LARGE" $LARGE_IP)
echo ""

# ============================================
# Download completed models
# ============================================
echo "Step 3: Downloading completed models..."

# Download Base
if [ "$BASE_STATUS" = "DONE" ]; then
  if [ ! -d "$OUTPUT_DIR/gpt2_base_prefix_682k" ]; then
    download_model "BASE" $BASE_IP "gpt2_base_prefix_682k"
  else
    echo "✓ BASE model already exists locally, skipping download"
  fi
else
  echo "⚠️  BASE training not complete yet, showing log:"
  show_log "BASE" $BASE_IP "training_base_prefix.log"
fi

# Download Medium
if [ "$MEDIUM_STATUS" = "DONE" ]; then
  if [ ! -d "$OUTPUT_DIR/gpt2_medium_prefix_682k" ]; then
    download_model "MEDIUM" $MEDIUM_IP "gpt2_medium_prefix_682k"
  else
    echo "✓ MEDIUM model already exists locally, skipping download"
  fi
else
  echo "⚠️  MEDIUM training not complete yet, showing log:"
  show_log "MEDIUM" $MEDIUM_IP "training_medium_prefix.log"
fi

# Download Large
if [ "$LARGE_STATUS" = "DONE" ]; then
  if [ ! -d "$OUTPUT_DIR/gpt2_large_prefix_682k" ]; then
    download_model "LARGE" $LARGE_IP "gpt2_large_prefix_682k"
  else
    echo "✓ LARGE model already exists locally, skipping download"
  fi
else
  echo "⚠️  LARGE training not complete yet, showing log:"
  show_log "LARGE" $LARGE_IP "training_large_prefix.log"
fi

echo ""

# ============================================
# Stop completed instances
# ============================================
echo "Step 4: Stopping completed instances..."

INSTANCES_TO_STOP=""

if [ "$BASE_STATUS" = "DONE" ] && [ -d "$OUTPUT_DIR/gpt2_base_prefix_682k" ]; then
  INSTANCES_TO_STOP="$INSTANCES_TO_STOP $BASE_ID"
  echo "  → Will stop BASE ($BASE_ID)"
fi

if [ "$MEDIUM_STATUS" = "DONE" ] && [ -d "$OUTPUT_DIR/gpt2_medium_prefix_682k" ]; then
  INSTANCES_TO_STOP="$INSTANCES_TO_STOP $MEDIUM_ID"
  echo "  → Will stop MEDIUM ($MEDIUM_ID)"
fi

if [ "$LARGE_STATUS" = "DONE" ] && [ -d "$OUTPUT_DIR/gpt2_large_prefix_682k" ]; then
  INSTANCES_TO_STOP="$INSTANCES_TO_STOP $LARGE_ID"
  echo "  → Will stop LARGE ($LARGE_ID)"
fi

if [ -n "$INSTANCES_TO_STOP" ]; then
  echo ""
  read -p "Stop these instances? (y/n) " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    aws ec2 stop-instances --instance-ids $INSTANCES_TO_STOP
    echo "✓ Instances stopped successfully"
  else
    echo "⚠️  Instances NOT stopped (user cancelled)"
  fi
else
  echo "  → No instances to stop"
fi

echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo ""
echo "Models downloaded:"
ls -lh $OUTPUT_DIR/gpt2_*_prefix_682k 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  None"
echo ""
echo "Still training:"
[ "$BASE_STATUS" != "DONE" ] && echo "  - BASE (use 'ssh -i $SSH_KEY ubuntu@$BASE_IP' to monitor)"
[ "$MEDIUM_STATUS" != "DONE" ] && echo "  - MEDIUM (use 'ssh -i $SSH_KEY ubuntu@$MEDIUM_IP' to monitor)"
[ "$LARGE_STATUS" != "DONE" ] && echo "  - LARGE (use 'ssh -i $SSH_KEY ubuntu@$LARGE_IP' to monitor)"
echo ""

# Check if ready for evaluation
if [ -d "$OUTPUT_DIR/gpt2_base_prefix_682k" ] && \
   [ -d "$OUTPUT_DIR/gpt2_medium_prefix_682k" ] && \
   [ -d "$OUTPUT_DIR/gpt2_large_prefix_682k" ]; then
  echo "✓ All models ready for evaluation!"
  echo ""
  echo "Next step: Run evaluation pipeline"
  echo "  bash run_all_evaluations.sh"
else
  echo "⚠️  Some models still training or not downloaded"
  echo ""
  echo "To monitor LARGE training:"
  echo "  ssh -i $SSH_KEY ubuntu@$LARGE_IP"
  echo "  tail -f ~/seriguela/training_large_prefix.log"
fi

echo ""
