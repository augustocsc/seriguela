#!/bin/bash
# Script to monitor evaluation progress and download results
# Usage: bash scripts/aws/monitor_evaluation.sh [PUBLIC_IP]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Get IP from argument or saved info
if [ -n "$1" ]; then
    PUBLIC_IP="$1"
else
    INFO_DIR="${HOME}/.seriguela"
    if [ -f "$INFO_DIR/last_evaluation_instance_ip.txt" ]; then
        PUBLIC_IP=$(cat "$INFO_DIR/last_evaluation_instance_ip.txt")
        print_status "Using saved IP: $PUBLIC_IP"
    else
        echo "Error: No IP provided and no saved IP found."
        echo "Usage: $0 <PUBLIC_IP>"
        exit 1
    fi
fi

# Get key name
INFO_DIR="${HOME}/.seriguela"
if [ -f "$INFO_DIR/last_evaluation_key_name.txt" ]; then
    KEY_NAME=$(cat "$INFO_DIR/last_evaluation_key_name.txt")
else
    KEY_NAME=$(aws ec2 describe-key-pairs --query "KeyPairs[0].KeyName" --output text 2>/dev/null)
fi

SSH_CMD="ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@${PUBLIC_IP}"

echo "=========================================="
echo "Monitoring Evaluation"
echo "=========================================="
echo "Instance: $PUBLIC_IP"
echo "Key: $KEY_NAME"
echo ""

# Check if setup is complete
print_status "Checking setup status..."
if $SSH_CMD 'test -f ~/.setup_complete'; then
    print_status "✅ Setup complete"
else
    print_warning "Setup still in progress. Waiting..."
    $SSH_CMD 'while [ ! -f ~/.setup_complete ]; do sleep 5; done; echo "Setup complete!"'
fi

echo ""
echo "=========================================="
echo "Evaluation Progress"
echo "=========================================="
echo "Press Ctrl+C to stop monitoring (evaluation will continue)"
echo ""

# Check if evaluation has started
if $SSH_CMD 'test -f ~/seriguela/evaluation_*.log'; then
    print_status "Evaluation in progress. Showing logs..."
    echo ""
    $SSH_CMD 'tail -f ~/seriguela/evaluation_*.log' || true
else
    print_warning "Evaluation hasn't started yet."
    echo ""
    echo "To start evaluation, run:"
    echo "  $SSH_CMD 'cd seriguela && source venv/bin/activate && bash scripts/aws/evaluate_models.sh'"
    echo ""
    echo "Or run in background:"
    echo "  $SSH_CMD 'cd seriguela && source venv/bin/activate && nohup bash scripts/aws/evaluate_models.sh > evaluation.log 2>&1 &'"
fi

echo ""
echo "=========================================="
echo "Download Results"
echo "=========================================="
echo ""

# Download results if available
if $SSH_CMD 'test -d ~/seriguela/evaluation_results/comparison'; then
    print_status "Downloading results..."

    # Create local directory
    mkdir -p ./evaluation_results/comparison

    # Download results
    scp -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no -r \
        ubuntu@${PUBLIC_IP}:~/seriguela/evaluation_results/comparison/* \
        ./evaluation_results/comparison/ 2>/dev/null || true

    # Download log files
    scp -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no \
        ubuntu@${PUBLIC_IP}:~/seriguela/evaluation_*.log \
        ./evaluation_results/ 2>/dev/null || true

    print_status "Results downloaded to: ./evaluation_results/"
    echo ""

    # Show latest comparison
    LATEST_COMPARISON=$(ls -t ./evaluation_results/comparison/comparison_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_COMPARISON" ]; then
        echo "Latest comparison results:"
        echo ""
        cat "$LATEST_COMPARISON" | jq '.comparison' 2>/dev/null || cat "$LATEST_COMPARISON"
    fi
else
    print_warning "No results available yet."
fi

echo ""
