#!/bin/bash
# Automatic Training Monitor and Notifier
# Monitors training process and runs analysis when complete

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} $1"; }
print_error() { echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $1"; }
print_header() { echo -e "\n${BLUE}========================================\n$1\n========================================${NC}\n"; }

# Configuration
PROJECT_DIR="/home/ubuntu/seriguela"
LOG_FILE="$HOME/training_success.log"
MONITOR_LOG="$HOME/monitor_output.log"
TRAINING_PID=""
CHECK_INTERVAL=60  # Check every 60 seconds
MODEL_PATH="./output/Se124M_700K_infix"
DATASET_REPO="augustocsc/sintetico_natural"
DATA_DIR="700K"
DATA_COLUMN="i_prompt_n"

cd "$PROJECT_DIR"
source venv/bin/activate

# Get training PID
get_training_pid() {
    TRAINING_PID=$(ps aux | grep "python scripts/train.py" | grep -v grep | awk '{print $2}')
}

# Check if training is running
is_training_running() {
    get_training_pid
    if [ -z "$TRAINING_PID" ]; then
        return 1
    else
        return 0
    fi
}

# Get training progress from log
get_progress() {
    if [ -f "$LOG_FILE" ]; then
        # Get last progress line
        tail -100 "$LOG_FILE" | grep -E "([0-9]+)%\|" | tail -1 | sed 's/.*\([0-9]\+\)%|.*/\1/' || echo "0"
    else
        echo "0"
    fi
}

# Get current epoch and step
get_training_stats() {
    if [ -f "$LOG_FILE" ]; then
        local last_line=$(tail -100 "$LOG_FILE" | grep -E "[0-9]+/21882" | tail -1)
        echo "$last_line"
    fi
}

# Send notification (multiple methods)
send_notification() {
    local title="$1"
    local message="$2"

    print_header "$title"
    echo "$message"

    # Save to notification file
    cat > "$HOME/training_notification.txt" << EOF
================================================================================
$title
$(date '+%Y-%m-%d %H:%M:%S')
================================================================================

$message

================================================================================
EOF

    print_status "Notification saved to: $HOME/training_notification.txt"
}

# Monitor training
print_header "Training Monitor Started"
print_status "Monitoring training process..."
print_status "Log file: $LOG_FILE"
print_status "Check interval: ${CHECK_INTERVAL}s"

START_TIME=$(date +%s)
LAST_PROGRESS=0

while true; do
    if is_training_running; then
        CURRENT_PROGRESS=$(get_progress)
        TRAINING_STATS=$(get_training_stats)

        # Show progress every check
        print_status "Training running (PID: $TRAINING_PID) - Progress: ${CURRENT_PROGRESS}%"

        if [ ! -z "$TRAINING_STATS" ]; then
            echo "         $TRAINING_STATS"
        fi

        # Check GPU
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
        echo "         GPU: $GPU_INFO"

        LAST_PROGRESS=$CURRENT_PROGRESS
        sleep $CHECK_INTERVAL
    else
        # Training finished or crashed
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))

        print_header "Training Process Ended"

        # Check if training completed successfully
        if grep -q "Training finished" "$LOG_FILE" 2>/dev/null || \
           grep -q "100%|" "$LOG_FILE" 2>/dev/null; then

            # SUCCESS - Training completed
            print_status "Training completed successfully!"
            print_status "Total time: ${HOURS}h ${MINUTES}m"

            # Extract final metrics
            FINAL_METRICS=$(tail -200 "$LOG_FILE" | grep -E "(train_loss|eval_loss)" | tail -5)

            send_notification "✅ Training Completed Successfully" \
"Training Duration: ${HOURS}h ${MINUTES}m
Model: GPT-2 (124M) with LoRA
Dataset: 700K infix
Output: $MODEL_PATH

Final Metrics:
$FINAL_METRICS

Wandb Dashboard:
https://wandb.ai/symbolic-gression/seriguela_700K_test

Starting automatic analysis...
"

            # Run automatic analysis
            print_header "Starting Automatic Analysis"
            bash "$PROJECT_DIR/scripts/aws/analyze_model.sh" "$MODEL_PATH" "$DATA_COLUMN" 2>&1 | tee "$HOME/analysis_output.log"

            print_status "Analysis complete! Check: $HOME/analysis_output.log"

        else
            # FAILED - Training crashed or was killed
            print_error "Training ended unexpectedly!"

            # Get last errors
            ERRORS=$(tail -50 "$LOG_FILE" | grep -E "(Error|Exception|Traceback)" | head -10)

            send_notification "❌ Training Failed or Interrupted" \
"Training Duration: ${HOURS}h ${MINUTES}m
Last Progress: ${LAST_PROGRESS}%

Possible Errors:
$ERRORS

Check full log: $LOG_FILE
"
        fi

        break
    fi
done

print_status "Monitor finished. Check notification file: $HOME/training_notification.txt"
