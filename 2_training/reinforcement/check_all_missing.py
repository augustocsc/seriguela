#!/usr/bin/env python3
"""
Check missing configs across all 6 AWS instances.

Usage:
    python check_all_missing.py
"""

import subprocess
import json
import time
import sys

# Instance IDs
INSTANCES = {
    'i-0ab8277c5128ef303': 'base_infix_n1',
    'i-0dcb39ad7278622ec': 'base_infix_n5',
    'i-00d7e518d26082914': 'base_infix_n9',
    'i-0aeeb70b76c5dc7d8': 'base_prefix_n1',
    'i-073564e75558da6f3': 'base_prefix_n5',
    'i-09aadd345995e5611': 'base_prefix_n9',
}

SSH_KEY = 'C:/Users/madeinweb/chave-gpu.pem'


def run_command(cmd):
    """Run shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def start_instances():
    """Start all 6 instances"""
    print("Starting all 6 instances...")
    instance_ids = ' '.join(INSTANCES.keys())
    cmd = f'aws ec2 start-instances --instance-ids {instance_ids}'
    stdout, stderr, _ = run_command(cmd)
    print("Instances starting...")


def stop_instances():
    """Stop all 6 instances"""
    print("\nStopping all instances...")
    instance_ids = ' '.join(INSTANCES.keys())
    cmd = f'aws ec2 stop-instances --instance-ids {instance_ids}'
    run_command(cmd)
    print("Instances stopped.")


def get_instance_ips():
    """Get public IPs for all instances"""
    instance_ids = ' '.join(INSTANCES.keys())
    cmd = f'aws ec2 describe-instances --instance-ids {instance_ids} --query "Reservations[*].Instances[*].[InstanceId,PublicIpAddress]" --output json'
    stdout, _, _ = run_command(cmd)
    data = json.loads(stdout)

    ip_map = {}
    for reservation in data:
        for instance in reservation:
            instance_id = instance[0]
            ip = instance[1]
            if ip:
                ip_map[instance_id] = ip

    return ip_map


def check_instance(ip, name, output_file):
    """Check missing configs on one instance"""
    print(f"\n{'='*60}")
    print(f"Checking {name} at {ip}")
    print('='*60)

    cmd = f'python check_missing_configs.py --ssh ubuntu@{ip} --key {SSH_KEY} --output {output_file}'
    stdout, stderr, returncode = run_command(cmd)

    print(stdout)
    if returncode != 0:
        print(f"Error: {stderr}", file=sys.stderr)


def main():
    try:
        # Start instances
        start_instances()

        # Wait for instances to be ready
        print("Waiting 40 seconds for instances to start...")
        time.sleep(40)

        # Get IPs
        print("Getting instance IPs...")
        ip_map = get_instance_ips()

        if len(ip_map) < 6:
            print(f"Warning: Only {len(ip_map)} instances have IPs. Some may not be running yet.")
            print("Waiting another 20 seconds...")
            time.sleep(20)
            ip_map = get_instance_ips()

        # Check each instance
        results = {}
        for instance_id, name in INSTANCES.items():
            if instance_id in ip_map:
                ip = ip_map[instance_id]
                output_file = f'missing_{name}.json'
                check_instance(ip, name, output_file)
                results[name] = output_file
            else:
                print(f"Warning: No IP found for {name} ({instance_id})")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        total_missing = 0
        for name, output_file in results.items():
            try:
                with open(output_file) as f:
                    missing = json.load(f)
                    count = len(missing)
                    total_missing += count
                    print(f"{name:20s}: {count:4d} missing")
            except:
                print(f"{name:20s}: ERROR reading results")

        print("-"*60)
        print(f"{'TOTAL':20s}: {total_missing:4d} missing")
        print(f"{'TARGET':20s}: 8,640 configs")
        print(f"{'COMPLETED':20s}: {8640 - total_missing:4d} ({(8640-total_missing)/8640*100:.2f}%)")

    finally:
        # Always stop instances
        stop_instances()


if __name__ == '__main__':
    main()
