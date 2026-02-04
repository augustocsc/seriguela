#!/usr/bin/env python3
import subprocess
import sys

wandb_key = input("Wandb API key: ").strip()
hf_token = input("HuggingFace token (Enter to skip): ").strip()

print("\nLaunching GPT-2 Medium training...")

cmd = ["bash", "scripts/aws/launch_medium_training.sh", "--wandb-key", wandb_key]
if hf_token:
    cmd.extend(["--hf-token", hf_token])

subprocess.run(cmd, check=True)
