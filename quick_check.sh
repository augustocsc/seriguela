#!/bin/bash
# Quick training status check

SSH_KEY="/c/Users/madeinweb/chave-gpu.pem"
BASE_IP="18.234.96.235"
MEDIUM_IP="34.229.252.142"
LARGE_IP="54.91.159.93"

echo "=========================================="
echo "Training Status Check - $(date '+%H:%M:%S')"
echo "=========================================="
echo ""

check_model() {
    local name=$1
    local ip=$2
    
    echo -n "$name: "
    
    # Check if model file exists
    STATUS=$(timeout 10 ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o LogLevel=ERROR ubuntu@$ip \
        'test -f ~/seriguela/output/gpt2_*/adapter_model.bin && echo "DONE" || echo "TRAINING"' 2>/dev/null || echo "ERROR")
    
    if [ "$STATUS" = "DONE" ]; then
        echo "✓ COMPLETED"
    elif [ "$STATUS" = "TRAINING" ]; then
        # Get last log line
        PROGRESS=$(timeout 10 ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o LogLevel=ERROR ubuntu@$ip \
            'tail -1 ~/training_*.log 2>/dev/null' 2>/dev/null | head -c 80)
        echo "⏳ Training... $PROGRESS"
    else
        echo "❌ ERROR or not accessible"
    fi
}

check_model "Base   (124M)" "$BASE_IP"
check_model "Medium (355M)" "$MEDIUM_IP"
check_model "Large  (774M)" "$LARGE_IP"

echo ""
echo "Instances status:"
aws ec2 describe-instances --instance-ids i-0855711efcac25a9c i-0eea77c3bbf1ea976 i-04dc6f51534d8185d \
    --query "Reservations[*].Instances[*].[Tags[?Key=='Model'].Value|[0],State.Name]" --output table 2>/dev/null

echo ""
echo "Run again: bash quick_check.sh"
