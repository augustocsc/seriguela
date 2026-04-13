#!/usr/bin/env python3
"""Download all Phase A results from AWS instances to local machine."""

import subprocess
import json
import time
from pathlib import Path

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

def start_instance(instance_id):
    """Start an EC2 instance."""
    print(f"Starting {instance_id}...")
    result = subprocess.run(
        f'aws ec2 start-instances --instance-ids {instance_id}',
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def get_instance_ip(instance_id):
    """Get public IP of instance."""
    result = subprocess.run(
        f'aws ec2 describe-instances --instance-ids {instance_id} '
        f'--query "Reservations[0].Instances[0].PublicIpAddress" --output text',
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None

def wait_for_ssh(ip, max_wait=300):
    """Wait for SSH to be available."""
    print(f"Waiting for SSH on {ip}...")
    start = time.time()
    while time.time() - start < max_wait:
        result = subprocess.run(
            f'ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=5 '
            f'ubuntu@{ip} "echo ready"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("SSH ready!")
            return True
        time.sleep(10)
    return False

def download_wandb_dir(ip, config_name):
    """Download wandb directory from instance."""
    local_path = LOCAL_DIR / config_name
    local_path.mkdir(exist_ok=True)

    print(f"\nDownloading wandb directory from {config_name}...")

    # Use rsync for efficient transfer
    result = subprocess.run(
        f'scp -i {SSH_KEY} -o StrictHostKeyChecking=no -r '
        f'ubuntu@{ip}:~/seriguela/2_training/reinforcement/wandb/ '
        f'{local_path}/',
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"Downloaded to {local_path}")
        return True
    else:
        print(f"Error: {result.stderr}")
        return False

def stop_instance(instance_id):
    """Stop an EC2 instance."""
    print(f"\nStopping {instance_id}...")
    result = subprocess.run(
        f'aws ec2 stop-instances --instance-ids {instance_id}',
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def main():
    print("="*60)
    print("DOWNLOADING PHASE A RESULTS FROM AWS")
    print("="*60)
    print(f"Will download from {len(INSTANCES)} instances")
    print(f"Local directory: {LOCAL_DIR.absolute()}")
    print("="*60)
    print()

    summary = []

    for config_name, instance_id in INSTANCES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {config_name}")
        print(f"Instance: {instance_id}")
        print(f"{'='*60}")

        # Start instance
        if not start_instance(instance_id):
            print(f"Failed to start {instance_id}")
            summary.append({'config': config_name, 'status': 'failed_start'})
            continue

        # Wait for instance to be running
        time.sleep(30)

        # Get IP
        ip = get_instance_ip(instance_id)
        if not ip:
            print(f"Failed to get IP for {instance_id}")
            summary.append({'config': config_name, 'status': 'failed_ip'})
            continue

        print(f"IP: {ip}")

        # Wait for SSH
        if not wait_for_ssh(ip):
            print(f"SSH not ready on {ip}")
            summary.append({'config': config_name, 'status': 'failed_ssh'})
            stop_instance(instance_id)
            continue

        # Download
        if download_wandb_dir(ip, config_name):
            summary.append({'config': config_name, 'status': 'success'})
        else:
            summary.append({'config': config_name, 'status': 'failed_download'})

        # Stop instance
        stop_instance(instance_id)

        # Wait before next instance
        time.sleep(10)

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
