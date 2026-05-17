#!/usr/bin/env python3
"""Download all Phase A results by zipping on server first (PARALLEL)."""

import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

INSTANCES = {
    'base_infix_n5': 'i-0dcb39ad7278622ec',
    'base_infix_n9': 'i-00d7e518d26082914',
    'base_prefix_n1': 'i-0aeeb70b76c5dc7d8',
    'base_prefix_n5': 'i-073564e75558da6f3',
    'base_prefix_n9': 'i-09aadd345995e5611',
}

SSH_KEY = 'C:/Users/madeinweb/chave-gpu.pem'
LOCAL_DIR = Path('.')

def run_cmd(cmd, timeout=300):
    """Run command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0, result.stdout.strip()

def download_one_instance(config_name, instance_id):
    """Download from one instance (runs in parallel)."""
    print(f"\n[{config_name}] Starting...")

    # Start instance
    print(f"[{config_name}] Starting instance...")
    success, _ = run_cmd(f'aws ec2 start-instances --instance-ids {instance_id}')
    if not success:
        return {'config': config_name, 'status': 'failed_start'}

    # Wait for instance
    time.sleep(45)

    # Get IP
    success, ip = run_cmd(
        f'aws ec2 describe-instances --instance-ids {instance_id} '
        f'--query "Reservations[0].Instances[0].PublicIpAddress" --output text'
    )
    if not success or not ip:
        return {'config': config_name, 'status': 'failed_ip'}

    print(f"[{config_name}] IP: {ip}")

    # Wait for SSH
    print(f"[{config_name}] Waiting for SSH...")
    for _ in range(30):
        success, _ = run_cmd(
            f'ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=5 '
            f'ubuntu@{ip} "echo ready"',
            timeout=10
        )
        if success:
            break
        time.sleep(10)
    else:
        return {'config': config_name, 'status': 'failed_ssh'}

    print(f"[{config_name}] SSH ready!")

    # Zip on server
    print(f"[{config_name}] Zipping wandb directory...")
    success, _ = run_cmd(
        f'ssh -i {SSH_KEY} -o StrictHostKeyChecking=no ubuntu@{ip} '
        f'"cd ~/seriguela/2_training/reinforcement && tar -czf wandb_{config_name}.tar.gz wandb/"',
        timeout=600
    )
    if not success:
        return {'config': config_name, 'status': 'failed_zip'}

    # Download zip
    print(f"[{config_name}] Downloading...")
    local_file = LOCAL_DIR / f"{config_name}_complete.tar.gz"
    success, _ = run_cmd(
        f'scp -i {SSH_KEY} -o StrictHostKeyChecking=no '
        f'ubuntu@{ip}:~/seriguela/2_training/reinforcement/wandb_{config_name}.tar.gz '
        f'{local_file}',
        timeout=600
    )
    if not success:
        return {'config': config_name, 'status': 'failed_download'}

    # Extract
    print(f"[{config_name}] Extracting...")
    extract_dir = LOCAL_DIR / f'wandb_{config_name}'
    extract_dir.mkdir(exist_ok=True)
    success, _ = run_cmd(
        f'cd {extract_dir} && tar -xzf ../{local_file.name}',
        timeout=300
    )

    # Stop instance
    print(f"[{config_name}] Stopping instance...")
    run_cmd(f'aws ec2 stop-instances --instance-ids {instance_id}')

    if success:
        print(f"[{config_name}] COMPLETE!")
        return {'config': config_name, 'status': 'success', 'file': str(local_file)}
    else:
        return {'config': config_name, 'status': 'failed_extract'}

def main():
    print("="*60)
    print("PARALLEL ZIP & DOWNLOAD FROM AWS")
    print("="*60)
    print(f"Downloading from {len(INSTANCES)} instances")
    print("="*60)
    print()

    # Download all in parallel
    summary = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_one_instance, config, instance_id): config
            for config, instance_id in INSTANCES.items()
        }

        for future in as_completed(futures):
            result = future.result()
            summary.append(result)

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for item in sorted(summary, key=lambda x: x['config']):
        status_sym = "OK" if item['status'] == 'success' else "X"
        print(f"[{status_sym}] {item['config']}: {item['status']}")

    successful = sum(1 for s in summary if s['status'] == 'success')
    print(f"\nTotal: {successful}/{len(INSTANCES)} successful")
    print("="*60)

if __name__ == '__main__':
    main()
