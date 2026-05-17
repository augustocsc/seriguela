#!/usr/bin/env python3
"""Download ALL results from W&B (no filter)."""

import wandb
import json
from pathlib import Path

# Setup
PROJECT = "symbolic-gression/seriguela"
LOCAL_DIR = Path('phase_a_results_wandb')
LOCAL_DIR.mkdir(exist_ok=True)

print("="*60)
print("DOWNLOADING ALL RUNS FROM W&B")
print("="*60)
print(f"Project: {PROJECT}")
print(f"Local directory: {LOCAL_DIR.absolute()}")
print("="*60)
print()

# Initialize API
api = wandb.Api(timeout=60)

# Get all runs (NO FILTER)
print("Fetching all runs...")
runs = list(api.runs(PROJECT))

print(f"Found {len(runs)} total runs")
print()

# Filter Phase A (base models) locally
phase_a_runs = []
for run in runs:
    try:
        model = run.config.get('model', '')
        if model and ('base_infix' in model or 'base_prefix' in model):
            phase_a_runs.append(run)
    except:
        pass

print(f"Phase A runs: {len(phase_a_runs)}")
print()

# Download
results = []
print("Downloading...")

for i, run in enumerate(phase_a_runs):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{len(phase_a_runs)}")

    try:
        data = {
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            'config': dict(run.config),
            'summary': dict(run.summary._json_dict),
            'created_at': str(run.created_at),
            'tags': run.tags,
        }

        results.append(data)

        # Save individual
        run_file = LOCAL_DIR / f"{run.id}.json"
        with open(run_file, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Error on {run.name}: {e}")

# Save summary
summary_file = LOCAL_DIR / "all_runs.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print("="*60)
print("DONE")
print("="*60)
print(f"Downloaded: {len(results)} Phase A runs")
print(f"Saved to: {LOCAL_DIR.absolute()}")
