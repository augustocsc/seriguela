#!/usr/bin/env python3
"""Download all Phase A results from AWS instances IN PARALLEL."""

import subprocess
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Instance configurations (from Phase A)
INSTANCES = {
    'base_infix_n1': 'i-0ab8277c5128ef303',
    'base_infix_n5': 'i-0dde26a3e50f1a7c8',
    'base_infix_n9': 'i-0e34fa5cdfc48c00c',
    'base_prefix_n1': 'i-0c3e7e7fa2c8e96fc',
    'base_prefix_n5': 'i-0dc6efb9e5ebf2e14',
    'base_prefix_n9': 'i-0f9cfb91b3fbf9597',
}

SSH_KEY = 'C:/Users/madeinweb/chave-gpu.pem'
LOCAL_DIR = Path('phase_a_results_raw')
LOCAL_DIR.mkdir(exist_ok=True)

def run_cmd(cmd):
    """Run command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip()

def start_all_instances():
    """Start all instances at once."""
    print("Starting all 6 instances...")
    instance_ids = ' '.join(INSTANCES.values())
    success, _ = run_cmd(f'aws ec2 start-instances --instance-ids {instance_ids}')
    if success:
        print("Started all instances!")
    return success

def get_instance_ip(instance_id):
    """Get public IP of instance."""
    success, ip = run_cmd(
        f'aws ec2 describe-instances --instance-ids {instance_id} '
        f'--query "Reservations[0].Instances[0].PublicIpAddress" --output text'
    )
    return ip if success else None

def wait_for_ssh(ip, max_wait=300):
    """Wait for SSH to be available."""
    start = time.time()
    while time.time() - start < max_wait:
        success, _ = run_cmd(
            f'ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=5 '
            f'ubuntu@{ip} "echo ready"'
        )
        if success:
            return True
        time.sleep(10)
    return False

def download_instance(config_name, instance_id):
    """Download from one instance (runs in parallel)."""
    print(f"\n[{config_name}] Starting download...")

    # Get IP
    ip = get_instance_ip(instance_id)
    if not ip:
        return {'config': config_name, 'status': 'failed_ip'}

    print(f"[{config_name}] IP: {ip}")

    # Wait for SSH
    if not wait_for_ssh(ip):
        return {'config': config_name, 'status': 'failed_ssh'}

    print(f"[{config_name}] SSH ready, downloading...")

    # Download
    local_path = LOCAL_DIR / config_name
    local_path.mkdir(exist_ok=True)

    success, _ = run_cmd(
        f'scp -i {SSH_KEY} -o StrictHostKeyChecking=no -r '
        f'ubuntu@{ip}:~/seriguela/2_training/reinforcement/wandb/ '
        f'{local_path}/'
    )

    if success:
        print(f"[{config_name}] Download complete!")
        return {'config': config_name, 'status': 'success'}
    else:
        return {'config': config_name, 'status': 'failed_download'}

def stop_all_instances():
    """Stop all instances at once."""
    print("\nStopping all instances...")
    instance_ids = ' '.join(INSTANCES.values())
    run_cmd(f'aws ec2 stop-instances --instance-ids {instance_ids}')
    print("Stop command sent")

def main():
    print("="*60)
    print("PARALLEL DOWNLOAD FROM AWS")
    print("="*60)
    print(f"Downloading from {len(INSTANCES)} instances IN PARALLEL")
    print(f"Local directory: {LOCAL_DIR.absolute()}")
    print("="*60)
    print()

    # Start all instances at once
    if not start_all_instances():
        print("Failed to start instances")
        return

    # Wait for instances to boot
    print("Waiting 45 seconds for instances to boot...")
    time.sleep(45)

    # Download from all instances in parallel
    summary = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(download_instance, config, instance_id): config
            for config, instance_id in INSTANCES.items()
        }

        for future in as_completed(futures):
            result = future.result()
            summary.append(result)
            print(f"\n*** {result['config']}: {result['status']} ***")

    # Stop all instances
    stop_all_instances()

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for item in summary:
        status_symbol = "OK" if item['status'] == 'success' else "X"
        print(f"[{status_symbol}] {item['config']}: {item['status']}")

    successful = sum(1 for s in summary if s['status'] == 'success')
    print(f"\nTotal: {successful}/{len(INSTANCES)} successful")
    print(f"Results saved to: {LOCAL_DIR.absolute()}")
    print("="*60)

if __name__ == '__main__':
    main()
