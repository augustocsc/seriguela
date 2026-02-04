#!/bin/bash
# Monitor Nguyen benchmarks progress on AWS

KEY_PATH="C:/Users/madeinweb/chave-gpu.pem"
INSTANCE_IP="54.91.123.196"

echo "=========================================="
echo "MONITORING NGUYEN BENCHMARKS"
echo "=========================================="
echo "Instance: $INSTANCE_IP"
echo "Time: $(date)"
echo ""

# Check if process is running
echo "Checking if process is running..."
PROCESS_COUNT=$(ssh -i "$KEY_PATH" ubuntu@$INSTANCE_IP \
  'ps aux | grep "run_all_nguyen_benchmarks.py" | grep -v grep | wc -l' 2>/dev/null)

if [ "$PROCESS_COUNT" -gt 0 ]; then
  echo "✓ Process is RUNNING ($PROCESS_COUNT processes)"
else
  echo "✗ Process NOT running (may have completed or failed)"
fi

echo ""
echo "=========================================="
echo "LATEST LOG OUTPUT"
echo "=========================================="
ssh -i "$KEY_PATH" ubuntu@$INSTANCE_IP 'tail -30 ~/seriguela/nguyen_benchmarks.log' 2>/dev/null

echo ""
echo "=========================================="
echo "PROGRESS CHECK"
echo "=========================================="

# Count completed experiments
COMPLETED=$(ssh -i "$KEY_PATH" ubuntu@$INSTANCE_IP \
  'grep -c "✓ Completed in" ~/seriguela/nguyen_benchmarks.log 2>/dev/null' 2>/dev/null || echo "0")

echo "Experiments completed: $COMPLETED / 36"

# Check for completion marker
COMPLETE_MARKER=$(ssh -i "$KEY_PATH" ubuntu@$INSTANCE_IP \
  'grep "SUITE COMPLETE" ~/seriguela/nguyen_benchmarks.log 2>/dev/null' 2>/dev/null)

if [ -n "$COMPLETE_MARKER" ]; then
  echo ""
  echo "🎉 =========================================="
  echo "🎉 ALL EXPERIMENTS COMPLETE!"
  echo "🎉 =========================================="
  echo ""
  echo "Download results:"
  echo "  scp -i \"$KEY_PATH\" -r ubuntu@$INSTANCE_IP:~/seriguela/results_nguyen_benchmarks ./"
  echo ""
  echo "STOP INSTANCE:"
  echo "  aws ec2 stop-instances --instance-ids i-07279c5889587abfe"
  echo ""
else
  echo ""
  echo "⏳ Experiments still running..."
  echo "   Estimated time remaining: ~$((3 * 60 - (COMPLETED * 5))) minutes"
  echo ""
  echo "Run this script again to check progress:"
  echo "  bash monitor_nguyen.sh"
fi

echo ""
echo "=========================================="
echo "QUICK COMMANDS"
echo "=========================================="
echo "SSH to instance:"
echo "  ssh -i \"$KEY_PATH\" ubuntu@$INSTANCE_IP"
echo ""
echo "Watch live log:"
echo "  ssh -i \"$KEY_PATH\" ubuntu@$INSTANCE_IP 'tail -f ~/seriguela/nguyen_benchmarks.log'"
echo ""
echo "Check GPU usage:"
echo "  ssh -i \"$KEY_PATH\" ubuntu@$INSTANCE_IP 'nvidia-smi'"
echo ""
