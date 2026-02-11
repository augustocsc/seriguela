#!/bin/bash
# Quick evaluation progress checker
# Usage: bash check_evaluation_progress.sh

SSH_KEY="C:/Users/madeinweb/chave-gpu.pem"
EVAL_IP="3.81.72.206"

echo "=============================================="
echo "Comprehensive Evaluation - Progress Check"
echo "=============================================="
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

echo "=== PROCESS STATUS ==="
ssh -i "$SSH_KEY" ubuntu@$EVAL_IP 'ps aux | grep -E "run_comprehensive|ppo_symbolic|grpo_symbolic" | grep -v grep | wc -l' | xargs -I {} echo "Active processes: {}"

echo ""
echo "=== GPU STATUS ==="
ssh -i "$SSH_KEY" ubuntu@$EVAL_IP 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits' | awk -F', ' '{print "  GPU Utilization: " $1 "%\n  VRAM Used: " $2 " MiB\n  Temperature: " $3 "°C"}'

echo ""
echo "=== LATEST LOG (last 15 lines) ==="
ssh -i "$SSH_KEY" ubuntu@$EVAL_IP 'tail -15 ~/seriguela/evaluation_complete.log'

echo ""
echo "=== COMPLETED EXPERIMENTS ==="
ssh -i "$SSH_KEY" ubuntu@$EVAL_IP 'find ~/seriguela/evaluation_results/20260211_143640 -name "summary.json" | wc -l' | xargs -I {} echo "Experiments completed: {}/96"

echo ""
echo "=== ESTIMATED COMPLETION ==="
COMPLETED=$(ssh -i "$SSH_KEY" ubuntu@$EVAL_IP 'find ~/seriguela/evaluation_results/20260211_143640 -name "summary.json" 2>/dev/null | wc -l')
TOTAL=96
if [ "$COMPLETED" -gt 0 ]; then
    PERCENT=$(awk "BEGIN {printf \"%.1f\", $COMPLETED * 100 / $TOTAL}")
    echo "Progress: $COMPLETED/$TOTAL experiments ($PERCENT%)"

    # Calculate estimated completion time (Windows-compatible)
    START_EPOCH=1739282200  # 2026-02-11 14:36:40 UTC
    CURRENT_EPOCH=$(date -u +%s 2>/dev/null || echo $(($(date +%s) + 10800)))  # Fallback for Windows
    ELAPSED=$((CURRENT_EPOCH - START_EPOCH))

    if [ "$COMPLETED" -gt 1 ] && [ "$ELAPSED" -gt 0 ]; then
        AVG_TIME=$(awk "BEGIN {printf \"%.0f\", $ELAPSED / $COMPLETED}")
        REMAINING=$((TOTAL - COMPLETED))
        ETA_SECONDS=$((AVG_TIME * REMAINING))
        echo "Average time per experiment: $((AVG_TIME / 60)) min $((AVG_TIME % 60)) sec"
        echo "Estimated time remaining: $((ETA_SECONDS / 3600))h $((ETA_SECONDS % 3600 / 60))min"

        ETA_EPOCH=$((CURRENT_EPOCH + ETA_SECONDS))
        echo "Estimated completion: ~$(awk "BEGIN {print strftime(\"%Y-%m-%d %H:%M UTC\", $ETA_EPOCH)}" 2>/dev/null || echo "22:30-23:00 UTC")"
    fi
else
    echo "Progress: Starting..."
fi

echo ""
echo "=============================================="
