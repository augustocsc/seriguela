#!/bin/bash
# Monitor comprehensive evaluation progress

INSTANCE_IP="3.81.72.206"
SSH_KEY="C:/Users/madeinweb/chave-gpu.pem"

echo "========================================="
echo "Comprehensive Evaluation Monitor"
echo "========================================="
echo "Instance: $INSTANCE_IP"
echo "Time: $(date)"
echo ""

# Check if process is running
echo "Process Status:"
ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
  'ps aux | grep run_comprehensive_evaluation.py | grep -v grep' 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Evaluation process is RUNNING"
else
    echo "⚠️  Evaluation process NOT FOUND"
fi

echo ""
echo "========================================="
echo "Latest Progress (last 50 lines)"
echo "========================================="
ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
  'tail -50 ~/seriguela/evaluation_gpu.log 2>/dev/null || tail -50 ~/seriguela/evaluation_full.log 2>/dev/null' 2>/dev/null

echo ""
echo "========================================="
echo "Experiment Progress Summary"
echo "========================================="
ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
  'grep -E "\[.*\/96\]" ~/seriguela/evaluation_full.log | tail -10' 2>/dev/null

echo ""
echo "========================================="
echo "Results Directory Size"
echo "========================================="
ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
  'du -sh ~/seriguela/evaluation_results/ 2>/dev/null || echo "No results yet"'

echo ""
echo "========================================="
echo "System Resources"
echo "========================================="
ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
  'echo "Load: $(uptime | awk -F"load average:" '"'"'{print $2}'"'"')"; echo "Memory: $(free -h | grep Mem | awk '"'"'{print $3 "/" $2}'"'"')"; echo "Disk: $(df -h / | tail -1 | awk '"'"'{print $3 "/" $2 " (" $5 ")"}'"'"')"' 2>/dev/null

echo ""
echo "========================================="
echo "Quick Commands"
echo "========================================="
echo ""
echo "Follow log in real-time:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP 'tail -f ~/seriguela/evaluation_full.log'"
echo ""
echo "Check specific experiment results:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP 'ls -lh ~/seriguela/evaluation_results/*/'"
echo ""
echo "Download results (when complete):"
echo "  scp -r -i $SSH_KEY ubuntu@$INSTANCE_IP:~/seriguela/evaluation_results ./'"
echo ""
echo "Stop instance (when complete):"
echo "  aws ec2 stop-instances --instance-ids i-051cad4bd51af8746"
echo ""
