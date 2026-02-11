#!/usr/bin/env python3
"""
Upload prefix dataset to HuggingFace Hub.
"""

import os
from datasets import load_from_disk
from huggingface_hub import login

# Read token from credentials file
tokens_file = os.path.expanduser("~/.tokens.txt")
if not os.path.exists(tokens_file):
    tokens_file = r"C:\Users\madeinweb\.tokens.txt"

hf_token = None
with open(tokens_file, 'r') as f:
    for line in f:
        if line.startswith('huggingface'):
            hf_token = line.split('=')[1].strip()
            break

if not hf_token:
    raise ValueError("HuggingFace token not found in tokens file")

print("Logging in to HuggingFace...")
login(token=hf_token)
print("[OK] Login successful!")

# Load dataset from disk
print("\nLoading dataset from disk...")
dataset_path = "./1_data/processed/700K_prefix_682k"
dataset = load_from_disk(dataset_path)

print(f"[OK] Loaded dataset:")
print(f"  - Train: {len(dataset['train']):,} examples")
print(f"  - Validation: {len(dataset['validation']):,} examples")
print(f"  - Total: {len(dataset['train']) + len(dataset['validation']):,} examples")

# Upload to Hub
print("\nUploading to HuggingFace Hub...")
repo_id = "augustocsc/sintetico_natural_prefix_682k"

dataset.push_to_hub(
    repo_id=repo_id,
    private=False,
    commit_message="Initial upload: 682K prefix notation dataset (train/validation split)"
)

print("\n" + "="*60)
print("[OK] Upload complete!")
print("="*60)
print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
print("")
print("Next steps:")
print("1. Add README.md to the repository")
print("   - Copy content from: 1_data/processed/DATASET_PREFIX_682K_README.md")
print("   - Upload via web interface or git")
print("")
print("2. Verify dataset loads correctly:")
print(f"   from datasets import load_dataset")
print(f"   ds = load_dataset('{repo_id}')")
print("="*60)
