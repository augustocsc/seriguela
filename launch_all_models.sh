#!/bin/bash
# Launch all 3 models (Base, Medium, Large) in parallel on AWS
# Part of Model Scaling Experiment (Feb 2025)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "Model Scaling Experiment - Parallel Launch"
echo "Training: GPT-2 Base, Medium, Large"
echo "=========================================="
echo ""

# Load credentials from ~/.tokens.txt
TOKENS_FILE="${HOME}/.tokens.txt"

if [ ! -f "$TOKENS_FILE" ]; then
    print_error "Credentials file not found: $TOKENS_FILE"
    print_error "Expected format:"
    print_error "  huggingface = hf_..."
    print_error "  wandb = wandb_v1_..."
    exit 1
fi

print_status "Loading credentials from $TOKENS_FILE"

WANDB_KEY=$(cat "$TOKENS_FILE" | grep -i wandb | cut -d= -f2 | tr -d ' ')
HF_TOKEN=$(cat "$TOKENS_FILE" | grep -i huggingface | cut -d= -f2 | tr -d ' ')

if [ -z "$WANDB_KEY" ]; then
    print_error "Wandb key not found in $TOKENS_FILE"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    print_warning "HuggingFace token not found. Models won't be pushed to Hub."
fi

print_status "Credentials loaded successfully"
echo ""

# Create log directory
LOG_DIR="./aws_launch_logs"
mkdir -p "$LOG_DIR"

print_status "Starting parallel instance launches..."
echo ""

# Launch Base (g5.xlarge) - 124M params, batch_size=8
print_status "[1/3] Launching Base (124M) on g5.xlarge..."
bash scripts/aws/launch_base_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > "$LOG_DIR/launch_base.log" 2>&1 &
BASE_PID=$!
print_status "    Base launch started (PID: $BASE_PID)"

# Stagger launches to avoid AWS rate limits
sleep 15

# Launch Medium (g5.xlarge) - 355M params, batch_size=4
print_status "[2/3] Launching Medium (355M) on g5.xlarge..."
bash scripts/aws/launch_medium_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > "$LOG_DIR/launch_medium.log" 2>&1 &
MEDIUM_PID=$!
print_status "    Medium launch started (PID: $MEDIUM_PID)"

# Stagger launches
sleep 15

# Launch Large (g5.2xlarge) - 774M params, batch_size=2
print_status "[3/3] Launching Large (774M) on g5.2xlarge..."
bash scripts/aws/launch_large_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > "$LOG_DIR/launch_large.log" 2>&1 &
LARGE_PID=$!
print_status "    Large launch started (PID: $LARGE_PID)"

echo ""
print_status "All launch processes initiated!"
echo ""
echo "Launch PIDs:"
echo "  Base:   $BASE_PID (log: $LOG_DIR/launch_base.log)"
echo "  Medium: $MEDIUM_PID (log: $LOG_DIR/launch_medium.log)"
echo "  Large:  $LARGE_PID (log: $LOG_DIR/launch_large.log)"
echo ""
print_status "Waiting for all launches to complete..."

# Wait for all launch processes
wait $BASE_PID
BASE_EXIT=$?
wait $MEDIUM_PID
MEDIUM_EXIT=$?
wait $LARGE_PID
LARGE_EXIT=$?

echo ""
echo "=========================================="
echo "Launch Results"
echo "=========================================="

# Check exit codes
FAILED=0

if [ $BASE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Base:   Launched successfully"
else
    echo -e "${RED}✗${NC} Base:   Failed (exit code: $BASE_EXIT)"
    FAILED=1
fi

if [ $MEDIUM_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Medium: Launched successfully"
else
    echo -e "${RED}✗${NC} Medium: Failed (exit code: $MEDIUM_EXIT)"
    FAILED=1
fi

if [ $LARGE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Large:  Launched successfully"
else
    echo -e "${RED}✗${NC} Large:  Failed (exit code: $LARGE_EXIT)"
    FAILED=1
fi

echo ""

if [ $FAILED -eq 0 ]; then
    print_status "All instances launched successfully!"
    echo ""

    # Show running instances
    print_status "Running training instances:"
    aws ec2 describe-instances \
      --filters "Name=tag:Name,Values=seriguela-*-training" "Name=instance-state-name,Values=running,pending" \
      --query "Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress,Tags[?Key=='Name'].Value|[0],Tags[?Key=='Model'].Value|[0]]" \
      --output table

    echo ""
    echo "=========================================="
    echo "Monitoring Instructions"
    echo "=========================================="
    echo ""
    echo "Check instance info files:"
    echo "  cat ~/.seriguela/base_instance_info.txt"
    echo "  cat ~/.seriguela/medium_instance_info.txt"
    echo "  cat ~/.seriguela/large_instance_info.txt"
    echo ""
    echo "Monitor Wandb dashboard:"
    echo "  https://wandb.ai/YOUR_USERNAME/seriguela"
    echo ""
    echo "Estimated training time:"
    echo "  Base:   2-3 hours"
    echo "  Medium: 3-4 hours"
    echo "  Large:  4-5 hours"
    echo ""
    echo "Estimated costs:"
    echo "  Base:   ~\$2-3 USD"
    echo "  Medium: ~\$3-4 USD"
    echo "  Large:  ~\$5-6 USD"
    echo "  Total:  ~\$10-13 USD"
    echo ""
    echo -e "${RED}CRITICAL: Remember to STOP instances after training!${NC}"
    echo ""
    echo "Stop all instances:"
    echo "  aws ec2 stop-instances --instance-ids \\"
    echo "    \$(aws ec2 describe-instances \\"
    echo "      --filters \"Name=tag:Name,Values=seriguela-*-training\" \"Name=instance-state-name,Values=running\" \\"
    echo "      --query \"Reservations[*].Instances[*].InstanceId\" \\"
    echo "      --output text)"
    echo ""

else
    print_error "Some instances failed to launch. Check logs:"
    if [ $BASE_EXIT -ne 0 ]; then
        echo "  Base:   $LOG_DIR/launch_base.log"
    fi
    if [ $MEDIUM_EXIT -ne 0 ]; then
        echo "  Medium: $LOG_DIR/launch_medium.log"
    fi
    if [ $LARGE_EXIT -ne 0 ]; then
        echo "  Large:  $LOG_DIR/launch_large.log"
    fi
    exit 1
fi
