#!/usr/bin/env python3
"""Monitor experiment and auto-stop instance when complete."""

import subprocess
import time
import sys

TARGET_RUNS = 1398  # 1368 original + 30 missing
INSTANCE_ID = 'i-0ab8277c5128ef303'
IP = '3.87.41.100'
SSH_KEY = 'C:/Users/madeinweb/chave-gpu.pem'

def get_run_count():
    """Get current run count via SSH"""
    try:
        result = subprocess.run(
            f'ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@{IP} '
            f'"ls ~/seriguela/2_training/reinforcement/wandb/ 2>/dev/null | grep -c run-"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception as e:
        print(f"Error getting run count: {e}")

    return None

def stop_instance():
    """Stop the EC2 instance"""
    print("\nStopping instance...")
    result = subprocess.run(
        f'aws ec2 stop-instances --instance-ids {INSTANCE_ID}',
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("Instance stopped successfully!")
        return True
    else:
        print(f"Error stopping instance: {result.stderr}")
        return False

def main():
    print("="*60)
    print("MONITORING EXPERIMENT")
    print("="*60)
    print(f"Instance: {INSTANCE_ID}")
    print(f"IP: {IP}")
    print(f"Target: {TARGET_RUNS} runs")
    print(f"Check interval: 2 minutes")
    print("="*60)
    print()

    check_num = 0

    while True:
        check_num += 1
        runs = get_run_count()

        if runs is not None:
            progress_pct = (runs / TARGET_RUNS) * 100
            remaining = TARGET_RUNS - runs

            print(f"Check #{check_num}: {runs}/{TARGET_RUNS} runs ({progress_pct:.1f}%) - {remaining} remaining")

            if runs >= TARGET_RUNS:
                print()
                print("="*60)
                print("ALL 30 MISSING CONFIGS COMPLETED!")
                print("="*60)

                if stop_instance():
                    print("\nDone! Instance stopped.")
                    sys.exit(0)
                else:
                    print("\nWarning: Could not stop instance automatically.")
                    print(f"Please stop manually: aws ec2 stop-instances --instance-ids {INSTANCE_ID}")
                    sys.exit(1)
        else:
            print(f"Check #{check_num}: Could not connect to instance (may be stopping)")

        # Wait 2 minutes
        time.sleep(120)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
        print(f"Instance {INSTANCE_ID} is still running!")
        print("Stop manually if needed: aws ec2 stop-instances --instance-ids {INSTANCE_ID}")
        sys.exit(0)
