#!/bin/bash
# Monitor complete evaluation and finetuning pipeline
# Usage: bash monitor_pipeline.sh

SSH_KEY="C:/Users/madeinweb/chave-gpu.pem"
EVAL_IP="100.31.69.18"
EVAL_ID="i-0bfa29e0a4e501d09"

echo "============================================"
echo "Pipeline Monitoring"
echo "============================================"
echo "Instance: $EVAL_ID ($EVAL_IP)"
echo ""

# Check if pipeline is running
echo "Checking if pipeline is running..."
RUNNING=$(ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$EVAL_IP \
  'ps aux | grep complete_evaluation_and_finetuning.py | grep -v grep' 2>/dev/null)

if [ -n "$RUNNING" ]; then
  echo "✓ Pipeline is RUNNING"
else
  echo "⚠️  Pipeline not running (may have completed or failed)"
fi

echo ""
echo "============================================"
echo "Latest Log Output (last 100 lines)"
echo "============================================"
ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$EVAL_IP \
  'tail -100 ~/seriguela/complete_pipeline.log' 2>/dev/null

echo ""
echo "============================================"
echo "Commands"
echo "============================================"
echo ""
echo "Follow log in real-time:"
echo "  ssh -i $SSH_KEY ubuntu@$EVAL_IP"
echo "  tail -f ~/seriguela/complete_pipeline.log"
echo ""
echo "Check results files:"
echo "  ssh ubuntu@$EVAL_IP 'ls -lh ~/seriguela/*.json'"
echo ""
echo "Download results when complete:"
echo "  scp -i $SSH_KEY ubuntu@$EVAL_IP:~/seriguela/*.json ./"
echo ""
echo "Re-run this monitor:"
echo "  bash monitor_pipeline.sh"
echo ""
