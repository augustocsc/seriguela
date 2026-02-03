#!/usr/bin/env python3
"""
Launch AWS instance for GPT-2 Medium training using configured credentials.
"""

import os
import sys
import subprocess
from pathlib import Path

def get_wandb_key():
    """Get Wandb API key from configured sources."""
    # Try environment variable
    key = os.environ.get('WANDB_API_KEY')
    if key:
        return key

    # Try wandb config file
    wandb_config = Path.home() / '.netrc'
    if wandb_config.exists():
        with open(wandb_config) as f:
            for line in f:
                if 'password' in line and 'api.wandb.ai' in f.read():
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]

    # Try alternative location
    wandb_config = Path.home() / '.config' / 'wandb' / 'settings'
    if wandb_config.exists():
        with open(wandb_config) as f:
            for line in f:
                if line.startswith('api_key'):
                    return line.split('=')[1].strip()

    return None

def get_hf_token():
    """Get HuggingFace token from configured sources."""
    # Try environment variable
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token

    # Try HF config file
    hf_config = Path.home() / '.huggingface' / 'token'
    if hf_config.exists():
        with open(hf_config) as f:
            return f.read().strip()

    return None

def main():
    print("="*60)
    print("Launching GPT-2 Medium Training on AWS")
    print("="*60)
    print()

    # Get credentials
    wandb_key = get_wandb_key()
    hf_token = get_hf_token()

    if not wandb_key:
        print("ERROR: Wandb API key not found!")
        print()
        print("Configure with:")
        print("  wandb login")
        print()
        print("Or set environment variable:")
        print("  export WANDB_API_KEY=your_key")
        sys.exit(1)

    print(f"✓ Wandb API key found: {wandb_key[:10]}...")

    if hf_token:
        print(f"✓ HuggingFace token found: {hf_token[:10]}...")
    else:
        print("⚠ HuggingFace token not found (optional)")
        print("  Model won't be pushed to Hub")

    print()
    print("Launching AWS instance...")
    print()

    # Launch via bash script
    script_dir = Path(__file__).parent.parent / 'scripts' / 'aws'
    launch_script = script_dir / 'launch_medium_training.sh'

    cmd = [
        'bash',
        str(launch_script),
        '--wandb-key', wandb_key,
    ]

    if hf_token:
        cmd.extend(['--hf-token', hf_token])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to launch instance: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
