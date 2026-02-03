#!/bin/bash
# Automatic Training Monitor - Checks completion and downloads models
# Prevents wasting $ on idle instances

set -e

# Configuration
INSTANCE_IDS="i-0855711efcac25a9c i-0eea77c3bbf1ea976 i-04dc6f51534d8185d"
SSH_KEY="/c/Users/madeinweb/chave-gpu.pem"
CHECK_INTERVAL=300  # 5 minutes

# IPs (current)
BASE_IP="18.234.96.235"
MEDIUM_IP="34.229.252.142"
LARGE_IP="54.91.159.93"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $1"; }

# Check if training is complete
check_complete() {
    local ip=$1
    local result=$(timeout 10 ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$ip \
        'test -f ~/seriguela/output/gpt2_*/adapter_model.bin && echo "DONE" || echo "TRAINING"' 2>/dev/null || echo "ERROR")
    echo "$result"
}

# Download model from instance
download_model() {
    local ip=$1
    local model_name=$2
    
    log "Downloading $model_name from $ip..."
    
    if scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r \
        ubuntu@$ip:~/seriguela/output/gpt2_*_700K_json ./output/ 2>&1; then
        log "✓ $model_name downloaded!"
        return 0
    else
        error "✗ Failed to download $model_name"
        return 1
    fi
}

log "=========================================="
log "Training Monitor Started"
log "=========================================="
log "Checking every $CHECK_INTERVAL seconds"
log ""

BASE_DONE=false
MEDIUM_DONE=false
LARGE_DONE=false
ITERATION=0

while true; do
    ((ITERATION++))
    log "=== Check #$ITERATION ==="
    
    ALL_DONE=true
    
    # Check Base
    if [ "$BASE_DONE" = false ]; then
        STATUS=$(check_complete "$BASE_IP")
        if [ "$STATUS" = "DONE" ]; then
            log "✓ Base COMPLETED!"
            download_model "$BASE_IP" "Base" && BASE_DONE=true
        else
            log "⏳ Base still training..."
            ALL_DONE=false
        fi
    fi
    
    # Check Medium
    if [ "$MEDIUM_DONE" = false ]; then
        STATUS=$(check_complete "$MEDIUM_IP")
        if [ "$STATUS" = "DONE" ]; then
            log "✓ Medium COMPLETED!"
            download_model "$MEDIUM_IP" "Medium" && MEDIUM_DONE=true
        else
            log "⏳ Medium still training..."
            ALL_DONE=false
        fi
    fi
    
    # Check Large
    if [ "$LARGE_DONE" = false ]; then
        STATUS=$(check_complete "$LARGE_IP")
        if [ "$STATUS" = "DONE" ]; then
            log "✓ Large COMPLETED!"
            download_model "$LARGE_IP" "Large" && LARGE_DONE=true
        else
            log "⏳ Large still training..."
            ALL_DONE=false
        fi
    fi
    
    # If all done, stop instances
    if [ "$ALL_DONE" = true ]; then
        log ""
        log "🎉 ALL MODELS COMPLETED!"
        log "Stopping instances..."
        
        aws ec2 stop-instances --instance-ids $INSTANCE_IDS
        
        log "✓ Instances stopping!"
        log "Models in: ./output/"
        
        echo "$(date): All done" > .monitor_complete
        exit 0
    fi
    
    log "Next check in $CHECK_INTERVAL seconds..."
    log ""
    sleep $CHECK_INTERVAL
done
