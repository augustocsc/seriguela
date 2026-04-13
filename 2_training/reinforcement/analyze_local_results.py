#!/usr/bin/env python3
"""Analyze locally downloaded Phase A results."""

import json
from pathlib import Path
import pandas as pd

WANDB_DIR = Path('wandb')

print("="*60)
print("ANALYZING LOCAL PHASE A RESULTS")
print("="*60)
print(f"Directory: {WANDB_DIR.absolute()}")
print()

# Find all runs
run_dirs = sorted(WANDB_DIR.glob('run-*'))
print(f"Found {len(run_dirs)} runs")
print()

# Parse each run
results = []
failed = []

print("Parsing run data...")
for i, run_dir in enumerate(run_dirs):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{len(run_dirs)}")

    try:
        # Read metadata
        metadata_file = run_dir / 'files' / 'wandb-metadata.json'
        summary_file = run_dir / 'files' / 'wandb-summary.json'

        if not metadata_file.exists() or not summary_file.exists():
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        with open(summary_file) as f:
            summary = json.load(f)

        # Extract config from args
        args = metadata.get('args', [])

        # Parse command line args
        config = {}
        for j in range(len(args)):
            if args[j].startswith('--'):
                key = args[j][2:]
                if j + 1 < len(args) and not args[j + 1].startswith('--'):
                    config[key] = args[j + 1]

        results.append({
            'run_name': run_dir.name,
            'model': config.get('model', ''),
            'problem': config.get('problem', ''),
            'algorithm': config.get('algorithm', ''),
            'reward': config.get('reward_type', ''),
            'penalty': config.get('penalty_type', ''),
            'temperature': config.get('temperature', ''),
            'prompt': config.get('prompt_type', ''),
            'noise': float(config.get('noise_level', 0)),
            'final_r2': summary.get('final_r2', 0),
            'best_r2': summary.get('best_r2', 0),
            'final_loss': summary.get('final_loss', 0),
            'valid_rate': summary.get('valid_expression_rate', 0),
        })

    except Exception as e:
        failed.append(run_dir.name)

print(f"\nParsed: {len(results)} runs")
print(f"Failed: {len(failed)} runs")
print()

# Create DataFrame
df = pd.DataFrame(results)

# Save to CSV
output_file = 'phase_a_base_infix_n1_results.csv'
df.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")
print()

# Summary statistics
print("="*60)
print("SUMMARY STATISTICS")
print("="*60)
print()

print("Configuration breakdown:")
print(f"  Models: {df['model'].unique().tolist()}")
print(f"  Problems: {df['problem'].unique().tolist()}")
print(f"  Algorithms: {df['algorithm'].nunique()} unique")
print(f"  Rewards: {df['reward'].nunique()} unique")
print(f"  Penalties: {df['penalty'].nunique()} unique")
print(f"  Temperatures: {df['temperature'].nunique()} unique")
print(f"  Prompts: {df['prompt'].nunique()} unique")
print(f"  Noise levels: {df['noise'].nunique()} unique")
print()

print("Performance metrics:")
print(f"  Avg final R²: {df['final_r2'].mean():.4f}")
print(f"  Best R² overall: {df['best_r2'].max():.4f}")
print(f"  Avg valid rate: {df['valid_rate'].mean():.4f}")
print()

print("Top 10 configurations by best R²:")
top10 = df.nlargest(10, 'best_r2')[['algorithm', 'reward', 'penalty', 'temperature', 'prompt', 'noise', 'best_r2']]
print(top10.to_string(index=False))
print()

print("="*60)
print("DONE")
print("="*60)
