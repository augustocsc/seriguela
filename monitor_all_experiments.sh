#!/bin/bash
# Monitor all 3 AWS instances running experiments

KEY="/c/Users/madeinweb/chave-gpu.pem"
IP1="3.90.154.4"      # eval-basic
IP2="23.20.79.242"    # nguyen-1-6
IP3="54.84.126.145"   # nguyen-7-12

echo "=========================================="
echo "Monitoring Seriguela Experiment Progress"
echo "=========================================="
date
echo ""

# Function to check if process is running
check_running() {
    local ip=$1
    local name=$2
    local pattern=$3

    count=$(ssh -i $KEY -o ConnectTimeout=10 ubuntu@$ip "ps aux | grep '$pattern' | grep -v grep | wc -l" 2>/dev/null)

    if [ $? -eq 0 ]; then
        if [ "$count" -gt 0 ]; then
            echo "[$name] ✓ $count processes running"
            return 0
        else
            echo "[$name] ✗ No processes running (completed or failed)"
            return 1
        fi
    else
        echo "[$name] ⚠ Connection failed"
        return 2
    fi
}

# Function to get log tail
show_progress() {
    local ip=$1
    local name=$2
    local logfile=$3

    echo ""
    echo "--- $name Progress ---"
    ssh -i $KEY -o ConnectTimeout=10 ubuntu@$ip "tail -5 $logfile 2>/dev/null" 2>/dev/null || echo "  (Log not available)"
}

echo "=== Instance 1: Quality Evaluations ==="
check_running $IP1 "Base" "evaluate_quality.*base"
check_running $IP1 "Medium" "evaluate_quality.*medium"
check_running $IP1 "Large" "evaluate_quality.*large"
show_progress $IP1 "Base Quality" "~/eval_base_quality.log"

echo ""
echo "=== Instance 2: Nguyen 1-6 ==="
check_running $IP2 "Nguyen 1-6" "run_nguyen_subset"
show_progress $IP2 "Nguyen 1-6" "~/nguyen_1_6.log"

echo ""
echo "=== Instance 3: Nguyen 7-12 ==="
check_running $IP3 "Nguyen 7-12" "run_nguyen_subset"
show_progress $IP3 "Nguyen 7-12" "~/nguyen_7_12.log"

echo ""
echo "=========================================="
echo "Check complete at $(date)"
echo "=========================================="
echo ""

# Check if any results are ready
echo "=== Checking for completed results ==="
for ip in $IP1 $IP2 $IP3; do
    name=$(ssh -i $KEY ubuntu@$ip 'hostname' 2>/dev/null)
    results=$(ssh -i $KEY ubuntu@$ip 'find ~/seriguela/results -name "*.json" -type f 2>/dev/null | wc -l' 2>/dev/null)
    if [ $? -eq 0 ] && [ "$results" -gt 0 ]; then
        echo "[$name] $results result files ready"
    fi
done

echo ""
echo "Run this script again to check progress:"
echo "  bash monitor_all_experiments.sh"
