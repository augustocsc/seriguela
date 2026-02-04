#!/usr/bin/env python3
"""
Simple launcher for GPT-2 Medium training on AWS.
Prompts for credentials and launches instance.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("GPT-2 Medium (355M) Training on AWS")
    print("="*60)
    print()

    # Get credentials
    wandb_key = input("Enter your Wandb API key: ").strip()
    if not wandb_key:
        print("ERROR: Wandb API key is required!")
        sys.exit(1)

    hf_token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()

    print()
    print("Launching AWS instance for GPT-2 Medium training...")
    print("- Instance: g5.xlarge (NVIDIA A10G, 24GB VRAM)")
    print("- Training: 3 epochs (~2-3 hours)")
    print("- Cost: ~$2-3 USD")
    print()

    # Launch
    script_path = Path(__file__).parent / 'scripts' / 'aws' / 'launch_medium_training.sh'

    cmd = ['bash', str(script_path), '--wandb-key', wandb_key]
    if hf_token:
        cmd.extend(['--hf-token', hf_token])

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("Instance launched successfully!")
        print("="*60)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to launch instance (exit code {e.returncode})")
        return 1
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return 1

if __name__ == '__main__':
    sys.exit(main())
