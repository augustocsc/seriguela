#!/bin/bash
# Monitor LARGE model training progress
# Created: 2026-02-10

SSH_KEY="C:/Users/madeinweb/chave-gpu.pem"
LARGE_IP="18.206.201.220"
LARGE_ID="i-060e3e00d1138c964"

echo "============================================"
echo "Monitoring LARGE Model Training"
echo "============================================"
echo "Instance: $LARGE_ID"
echo "IP: $LARGE_IP"
echo ""

# Check if instance is running
echo "Checking instance status..."
STATUS=$(aws ec2 describe-instances \
  --instance-ids $LARGE_ID \
  --query "Reservations[0].Instances[0].State.Name" \
  --output text)

echo "Instance state: $STATUS"
echo ""

if [ "$STATUS" != "running" ]; then
  echo "⚠️  Instance is not running! Cannot monitor."
  exit 1
fi

# Check training completion
echo "Checking training completion..."
COMPLETE=$(ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$LARGE_IP \
  'test -f ~/.training_complete && echo "YES" || echo "NO"' 2>/dev/null)

if [ "$COMPLETE" = "YES" ]; then
  echo "✓ Training COMPLETE!"
  echo ""
  echo "To download the model:"
  echo "  scp -i $SSH_KEY -r ubuntu@$LARGE_IP:~/seriguela/output/gpt2_large_prefix_682k ./output/"
  echo ""
  echo "To stop the instance:"
  echo "  aws ec2 stop-instances --instance-ids $LARGE_ID"
  echo ""
  exit 0
fi

echo "Training still in progress..."
echo ""

# Show training progress
echo "============================================"
echo "Training Log (last 50 lines)"
echo "============================================"
ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$LARGE_IP \
  'tail -50 ~/seriguela/training_large_prefix.log' 2>/dev/null

echo ""
echo "============================================"
echo "GPU Usage"
echo "============================================"
ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$LARGE_IP \
  'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv' 2>/dev/null

echo ""
echo "============================================"
echo "Estimated Time Remaining"
echo "============================================"

# Extract current step and total steps from log
PROGRESS=$(ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$LARGE_IP \
  "grep -o '[0-9]*/[0-9]*' ~/seriguela/training_large_prefix.log | tail -1" 2>/dev/null)

if [ -n "$PROGRESS" ]; then
  CURRENT=$(echo $PROGRESS | cut -d'/' -f1)
  TOTAL=$(echo $PROGRESS | cut -d'/' -f2)
  PERCENT=$(echo "scale=2; $CURRENT * 100 / $TOTAL" | bc)
  REMAINING=$((TOTAL - CURRENT))

  echo "Progress: $CURRENT / $TOTAL steps ($PERCENT%)"
  echo "Remaining steps: $REMAINING"

  # Extract speed (it/s)
  SPEED=$(ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$LARGE_IP \
    "grep -oP '[0-9.]+it/s' ~/seriguela/training_large_prefix.log | tail -1 | cut -d'i' -f1" 2>/dev/null)

  if [ -n "$SPEED" ]; then
    TIME_REMAINING=$(echo "scale=2; $REMAINING / $SPEED / 3600" | bc)
    echo "Speed: $SPEED it/s"
    echo "Estimated time remaining: ${TIME_REMAINING} hours"
  fi
else
  echo "Could not extract progress information"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo ""
echo "Follow training log in real-time:"
echo "  ssh -i $SSH_KEY ubuntu@$LARGE_IP"
echo "  tail -f ~/seriguela/training_large_prefix.log"
echo ""
echo "Check completion:"
echo "  ssh -i $SSH_KEY ubuntu@$LARGE_IP 'test -f ~/.training_complete && echo DONE || echo Running'"
echo ""
echo "Re-run this monitor:"
echo "  bash monitor_large_training.sh"
echo ""
