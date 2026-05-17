#!/usr/bin/env python3
"""Download all Phase A results from HuggingFace to local machine."""

import os
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
import pandas as pd

# Setup paths
RESULTS_DIR = Path("phase_a_results")
RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("DOWNLOADING PHASE A RESULTS FROM HUGGINGFACE")
print("="*60)
print()

# Download dataset
print("1. Loading dataset from HuggingFace...")
print("   Repository: augustocsc/seriguela-results")
print()

try:
    dataset = load_dataset("augustocsc/seriguela-results", split="train")
    print(f"✓ Loaded {len(dataset)} results")
    print()
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print()
    print("Trying alternative method with HfApi...")

    # Alternative: download files directly
    api = HfApi()
    files = api.list_repo_files("augustocsc/seriguela-results", repo_type="dataset")
    print(f"Found {len(files)} files in repository")

    for file in files:
        if file.endswith('.json') or file.endswith('.parquet'):
            print(f"Downloading {file}...")
            api.hf_hub_download(
                repo_id="augustocsc/seriguela-results",
                filename=file,
                repo_type="dataset",
                local_dir=RESULTS_DIR
            )

    print("\n✓ Files downloaded directly")
    exit(0)

# Filter Phase A results (Base models only)
print("2. Filtering Phase A results (Base models)...")
base_models = ["base_infix", "base_prefix"]
phase_a = dataset.filter(lambda x: x.get('model', '').startswith('base_'))
print(f"✓ Found {len(phase_a)} Phase A results")
print()

# Organize by model and problem
print("3. Organizing results by model and problem...")
results_by_config = {}

for result in phase_a:
    model = result.get('model', 'unknown')
    problem = result.get('problem', 'unknown')
    key = f"{model}_{problem}"

    if key not in results_by_config:
        results_by_config[key] = []

    results_by_config[key].append(result)

print(f"✓ Organized into {len(results_by_config)} configurations:")
for key, results in results_by_config.items():
    print(f"   - {key}: {len(results)} results")
print()

# Save each configuration to separate files
print("4. Saving results to local files...")
for key, results in results_by_config.items():
    output_file = RESULTS_DIR / f"{key}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved {len(results)} results to {output_file}")

print()

# Create summary CSV
print("5. Creating summary CSV...")
summary_data = []

for result in phase_a:
    summary_data.append({
        'model': result.get('model'),
        'problem': result.get('problem'),
        'algorithm': result.get('algorithm'),
        'reward': result.get('reward_type'),
        'penalty': result.get('penalty_type'),
        'temperature': result.get('temperature'),
        'prompt': result.get('prompt_type'),
        'noise': result.get('noise_level'),
        'final_r2': result.get('final_r2'),
        'final_loss': result.get('final_loss'),
        'best_r2': result.get('best_r2'),
        'convergence_epoch': result.get('convergence_epoch'),
        'valid_rate': result.get('valid_expression_rate'),
    })

df = pd.DataFrame(summary_data)
summary_file = RESULTS_DIR / "phase_a_summary.csv"
df.to_csv(summary_file, index=False)
print(f"✓ Saved summary to {summary_file}")
print()

# Statistics
print("6. Summary Statistics:")
print(f"   Total results: {len(phase_a)}")
print(f"   Models: {df['model'].nunique()}")
print(f"   Problems: {df['problem'].nunique()}")
print(f"   Algorithms: {df['algorithm'].nunique()}")
print(f"   Unique configs: {len(results_by_config)}")
print()
print(f"   Avg final R²: {df['final_r2'].mean():.4f}")
print(f"   Avg valid rate: {df['valid_rate'].mean():.4f}")
print()

print("="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
print()
print("Files created:")
print(f"  - {len(results_by_config)} JSON files (one per model-problem)")
print(f"  - 1 CSV summary file")
print()
