#!/usr/bin/env python3
"""Analyze ALL Phase A results (all 6 configs)."""

import json
from pathlib import Path
import pandas as pd

# All wandb directories
WANDB_DIRS = {
    'base_infix_n1': Path('wandb'),
    'base_infix_n5': Path('wandb_base_infix_n5/wandb'),
    'base_infix_n9': Path('wandb_base_infix_n9/wandb'),
    'base_prefix_n1': Path('wandb_base_prefix_n1/wandb'),
    'base_prefix_n5': Path('wandb_base_prefix_n5/wandb'),
    'base_prefix_n9': Path('wandb_base_prefix_n9/wandb'),
}

print("="*60)
print("ANALYZING ALL PHASE A RESULTS")
print("="*60)
print()

all_results = []
config_stats = {}

# Parse each configuration
for config_name, wandb_dir in WANDB_DIRS.items():
    print(f"Processing {config_name}...")

    if not wandb_dir.exists():
        print(f"  WARNING: {wandb_dir} not found!")
        continue

    run_dirs = sorted(wandb_dir.glob('run-*'))
    print(f"  Found {len(run_dirs)} runs")

    config_results = []

    for run_dir in run_dirs:
        try:
            metadata_file = run_dir / 'files' / 'wandb-metadata.json'
            summary_file = run_dir / 'files' / 'wandb-summary.json'

            if not metadata_file.exists() or not summary_file.exists():
                continue

            with open(metadata_file) as f:
                metadata = json.load(f)

            with open(summary_file) as f:
                summary = json.load(f)

            # Parse args
            args = metadata.get('args', [])
            config = {}
            for j in range(len(args)):
                if args[j].startswith('--'):
                    key = args[j][2:]
                    if j + 1 < len(args) and not args[j + 1].startswith('--'):
                        config[key] = args[j + 1]

            result = {
                'config': config_name,
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
            }

            config_results.append(result)
            all_results.append(result)

        except Exception as e:
            continue

    config_stats[config_name] = {
        'total_runs': len(config_results),
        'avg_best_r2': sum(r['best_r2'] for r in config_results) / len(config_results) if config_results else 0,
        'max_best_r2': max((r['best_r2'] for r in config_results), default=0),
    }

    print(f"  Parsed: {len(config_results)} runs")

print()
print(f"Total results: {len(all_results)}")
print()

# Create DataFrame
df = pd.DataFrame(all_results)

# Save complete results
output_file = 'phase_a_all_results.csv'
df.to_csv(output_file, index=False)
print(f"Saved complete results to: {output_file}")
print()

# Summary by config
print("="*60)
print("SUMMARY BY CONFIGURATION")
print("="*60)
for config, stats in config_stats.items():
    print(f"\n{config}:")
    print(f"  Runs: {stats['total_runs']}")
    print(f"  Avg best R²: {stats['avg_best_r2']:.4f}")
    print(f"  Max best R²: {stats['max_best_r2']:.4f}")

print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)
print()

print("Breakdown:")
print(f"  Total runs: {len(df)}")
print(f"  Models: {df['model'].nunique()} unique")
print(f"  Problems: {df['problem'].nunique()} unique")
print(f"  Algorithms: {df['algorithm'].nunique()} unique")
print(f"  Configs per model-problem: {len(df) / 6:.0f}")
print()

print("Performance:")
print(f"  Avg final R²: {df['final_r2'].mean():.4f}")
print(f"  Avg best R²: {df['best_r2'].mean():.4f}")
print(f"  Best R² overall: {df['best_r2'].max():.4f}")
print(f"  Avg valid rate: {df['valid_rate'].mean():.4f}")
print()

# Best configurations overall
print("="*60)
print("TOP 10 CONFIGURATIONS (by best R²)")
print("="*60)
top10 = df.nlargest(10, 'best_r2')[
    ['config', 'problem', 'algorithm', 'reward', 'penalty',
     'temperature', 'prompt', 'noise', 'best_r2']
]
print(top10.to_string(index=False))
print()

# Best by algorithm
print("="*60)
print("BEST PERFORMANCE BY ALGORITHM")
print("="*60)
for algo in sorted(df['algorithm'].unique()):
    algo_df = df[df['algorithm'] == algo]
    best = algo_df.nlargest(1, 'best_r2').iloc[0]
    print(f"\n{algo}:")
    print(f"  Best R²: {best['best_r2']:.4f}")
    print(f"  Config: {best['config']} - {best['problem']}")
    print(f"  Settings: temp={best['temperature']}, prompt={best['prompt']}, noise={best['noise']}")

print("\n" + "="*60)
print("DONE")
print("="*60)
