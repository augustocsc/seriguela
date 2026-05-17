#!/usr/bin/env python3
"""Download all Phase A results from W&B."""

import wandb
import json
from pathlib import Path
from tqdm import tqdm

# Setup
PROJECT = "symbolic-gression/seriguela"
LOCAL_DIR = Path('phase_a_results_wandb')
LOCAL_DIR.mkdir(exist_ok=True)

print("="*60)
print("DOWNLOADING PHASE A RESULTS FROM W&B")
print("="*60)
print(f"Project: {PROJECT}")
print(f"Local directory: {LOCAL_DIR.absolute()}")
print("="*60)
print()

# Initialize API
api = wandb.Api(timeout=60)

# Get all runs
print("Fetching all runs from W&B...")
runs = api.runs(PROJECT, filters={
    "config.model": {"$in": ["base_infix", "base_prefix"]}
})

print(f"Found {len(runs)} Phase A runs")
print()

# Download each run's data
results = []
failed = []

print("Downloading run data...")
for i, run in enumerate(tqdm(runs)):
    try:
        # Extract key data
        data = {
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            'config': dict(run.config),
            'summary': dict(run.summary),
            'history_keys': list(run.history().columns) if hasattr(run, 'history') else []
        }

        results.append(data)

        # Save individual run
        run_file = LOCAL_DIR / f"{run.name}_{run.id}.json"
        with open(run_file, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"\nError on {run.name}: {e}")
        failed.append({'name': run.name, 'error': str(e)})
        continue

# Save summary
summary_file = LOCAL_DIR / "all_runs_summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print("="*60)
print("DOWNLOAD COMPLETE")
print("="*60)
print(f"Downloaded: {len(results)} runs")
print(f"Failed: {len(failed)} runs")
print(f"Saved to: {LOCAL_DIR.absolute()}")
print()

if failed:
    print("Failed runs:")
    for item in failed[:10]:
        print(f"  - {item['name']}: {item['error']}")
