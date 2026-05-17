#!/usr/bin/env python3
"""
Evaluate Phase A results and identify what's missing.

This script:
1. Downloads results from HuggingFace
2. Analyzes which hyperparameter combinations work best
3. Identifies gaps in coverage
4. Recommends configs for Phase B

Usage:
    python evaluate_phase_a_results.py
    python evaluate_phase_a_results.py --download-from-instances
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from huggingface_hub import HfApi, hf_hub_download


def download_results_from_hf(limit=None):
    """Download all Base model results from HuggingFace"""
    print("Downloading results from HuggingFace...")

    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    api = HfApi()
    files = list(api.list_repo_files('augustocsc/seriguela-results', repo_type='dataset'))

    # Filter for base model results
    base_files = [f for f in files if ('base_infix' in f or 'base_prefix' in f) and f.endswith('.json')]

    if limit:
        base_files = base_files[:limit]

    print(f"Found {len(base_files)} result files")

    results = []
    errors = 0

    for i, file in enumerate(base_files):
        if i % 100 == 0:
            print(f"  Downloading {i}/{len(base_files)}...")

        try:
            path = hf_hub_download(
                repo_id='augustocsc/seriguela-results',
                filename=file,
                repo_type='dataset'
            )

            with open(path) as f:
                data = json.load(f)

            # Extract key info
            result = {
                'algorithm': data.get('algorithm'),
                'model': data.get('model'),
                'problem': data.get('problem'),
                'config': data.get('config', {}),
                'best_r2': data.get('best_r2'),
                'test_r2': data.get('test_r2'),
                'test_mse': data.get('test_mse'),
                'test_valid': data.get('test_valid'),
                'generalization_gap': data.get('generalization_gap'),
                'best_expression': data.get('best_expression'),
                'total_steps': data.get('total_steps'),
                'reward_fn': str(data.get('reward_fn', ''))[:50],
                'penalty_strategy': str(data.get('penalty_strategy', ''))[:50],
                'temp_scheduler': str(data.get('temp_scheduler', ''))[:50],
            }

            results.append(result)

        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"    Error downloading {file}: {e}")

    print(f"\nDownloaded {len(results)} results ({errors} errors)")
    return results


def download_results_from_instances():
    """Download results from all 6 AWS instances via SSH"""
    print("Downloading results from AWS instances...")
    print("This requires instances to be running!")

    # Instance IDs and names
    instances = {
        'i-0ab8277c5128ef303': 'base_infix_n1',
        'i-0dcb39ad7278622ec': 'base_infix_n5',
        'i-00d7e518d26082914': 'base_infix_n9',
        'i-0aeeb70b76c5dc7d8': 'base_prefix_n1',
        'i-073564e75558da6f3': 'base_prefix_n5',
        'i-09aadd345995e5611': 'base_prefix_n9',
    }

    # Start instances
    print("Starting instances...")
    instance_ids = ' '.join(instances.keys())
    subprocess.run(f'aws ec2 start-instances --instance-ids {instance_ids}', shell=True)

    print("Waiting 40 seconds...")
    import time
    time.sleep(40)

    # Get IPs
    result = subprocess.run(
        f'aws ec2 describe-instances --instance-ids {instance_ids} --query "Reservations[*].Instances[*].[InstanceId,PublicIpAddress]" --output json',
        shell=True, capture_output=True, text=True
    )

    data = json.loads(result.stdout)
    ip_map = {}
    for reservation in data:
        for instance in reservation:
            ip_map[instance[0]] = instance[1]

    # Download from each
    all_results = []

    for instance_id, name in instances.items():
        if instance_id not in ip_map:
            print(f"Warning: No IP for {name}")
            continue

        ip = ip_map[instance_id]
        print(f"\nDownloading from {name} ({ip})...")

        # Create remote script to extract all results
        remote_script = """
import os
import json

wandb_dir = os.path.expanduser('~/seriguela/2_training/reinforcement/wandb')
runs = [d for d in os.listdir(wandb_dir) if d.startswith('run-')]

results = []
for run_dir in runs:
    # Try to find results file
    results_file = os.path.join(wandb_dir, run_dir, 'files', 'results.json')
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                data = json.load(f)
                results.append(data)
        except:
            pass

print(json.dumps(results))
"""

        ssh_cmd = f'ssh -i C:/Users/madeinweb/chave-gpu.pem -o StrictHostKeyChecking=no ubuntu@{ip} "python3 -c \\"{remote_script}\\"" 2>/dev/null'

        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and result.stdout:
            try:
                instance_results = json.loads(result.stdout)
                print(f"  Got {len(instance_results)} results")
                all_results.extend(instance_results)
            except:
                print(f"  Error parsing JSON")
        else:
            print(f"  SSH failed")

    # Stop instances
    print("\nStopping instances...")
    subprocess.run(f'aws ec2 stop-instances --instance-ids {instance_ids}', shell=True)

    return all_results


def analyze_hyperparameter_impact(results):
    """Analyze which hyperparameters have the most impact"""
    print("\n" + "="*60)
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print("="*60)

    # Group by hyperparameter
    by_algo = defaultdict(list)
    by_reward = defaultdict(list)
    by_penalty = defaultdict(list)
    by_temp = defaultdict(list)
    by_prompt = defaultdict(list)
    by_problem = defaultdict(list)
    by_model = defaultdict(list)

    for r in results:
        if r['test_r2'] is not None:
            algo = r['algorithm']
            reward = r['config'].get('reward_type', 'unknown')
            penalty = r['config'].get('penalty_type', 'unknown')
            temp = r['config'].get('temperature_schedule', 'unknown')
            prompt = r['config'].get('prompt_type', 'unknown')
            problem = r['problem']
            model = r['model']

            by_algo[algo].append(r['test_r2'])
            by_reward[reward].append(r['test_r2'])
            by_penalty[penalty].append(r['test_r2'])
            by_temp[temp].append(r['test_r2'])
            by_prompt[prompt].append(r['test_r2'])
            by_problem[problem].append(r['test_r2'])
            by_model[model].append(r['test_r2'])

    def print_stats(name, data_dict):
        print(f"\n{name}:")
        items = []
        for key, values in data_dict.items():
            if len(values) > 0:
                mean_r2 = np.mean(values)
                std_r2 = np.std(values)
                median_r2 = np.median(values)
                count = len(values)
                items.append((key, mean_r2, std_r2, median_r2, count))

        items.sort(key=lambda x: x[1], reverse=True)

        print(f"  {'Name':<20} {'Mean R²':<10} {'Std':<10} {'Median R²':<10} {'Count':<8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for name, mean, std, median, count in items:
            print(f"  {name:<20} {mean:>10.4f} {std:>10.4f} {median:>10.4f} {count:>8d}")

    print_stats("Algorithm", by_algo)
    print_stats("Reward", by_reward)
    print_stats("Penalty", by_penalty)
    print_stats("Temperature", by_temp)
    print_stats("Prompt", by_prompt)
    print_stats("Problem", by_problem)
    print_stats("Model", by_model)


def find_best_configs(results, top_k=10):
    """Find top-K best performing configs"""
    print("\n" + "="*60)
    print(f"TOP {top_k} CONFIGURATIONS")
    print("="*60)

    # Sort by test R²
    sorted_results = sorted(
        [r for r in results if r['test_r2'] is not None],
        key=lambda x: x['test_r2'],
        reverse=True
    )[:top_k]

    for i, r in enumerate(sorted_results, 1):
        print(f"\n{i}. Test R² = {r['test_r2']:.4f} | Gen Gap = {r.get('generalization_gap', 'N/A')}")
        print(f"   Model: {r['model']}")
        print(f"   Problem: {r['problem']}")
        print(f"   Algorithm: {r['algorithm']}")
        print(f"   Reward: {r['config'].get('reward_type', 'unknown')}")
        print(f"   Penalty: {r['config'].get('penalty_type', 'unknown')}")
        print(f"   Temperature: {r['config'].get('temperature_schedule', 'unknown')}")
        print(f"   Prompt: {r['config'].get('prompt_type', 'unknown')}")
        if r['best_expression']:
            expr = r['best_expression']
            if len(expr) > 80:
                expr = expr[:77] + "..."
            print(f"   Expression: {expr}")

    return sorted_results[:top_k]


def identify_gaps(results):
    """Identify which hyperparameter combinations are under-tested"""
    print("\n" + "="*60)
    print("COVERAGE GAPS")
    print("="*60)

    # Count configs by combination
    combos = defaultdict(int)

    for r in results:
        algo = r['algorithm']
        reward = r['config'].get('reward_type', 'unknown')
        penalty = r['config'].get('penalty_type', 'unknown')

        combo = f"{algo}_{reward}_{penalty}"
        combos[combo] += 1

    # Expected: 5 algos × 3 rewards × 2 penalties = 30 combos
    # × 4 temps × 3 prompts × 4 noise × 6 model-problem = 8,640

    print("\nCombination coverage (algo × reward × penalty):")
    print(f"  {'Combination':<40} {'Count':<8}")
    print(f"  {'-'*40} {'-'*8}")

    for combo, count in sorted(combos.items(), key=lambda x: x[1]):
        print(f"  {combo:<40} {count:>8d}")

    # Find missing combos
    expected_algos = ['best_of_n', 'bon_ppo', 'bon_grpo', 'pure_ppo', 'pure_grpo']
    expected_rewards = ['length_penalized', 'r2_clipped', 'sr_ic']
    expected_penalties = ['binary', 'gradient']

    all_expected = set()
    for algo in expected_algos:
        for reward in expected_rewards:
            for penalty in expected_penalties:
                all_expected.add(f"{algo}_{reward}_{penalty}")

    tested = set(combos.keys())
    missing = all_expected - tested

    if missing:
        print(f"\nMissing combinations: {len(missing)}")
        for m in sorted(missing):
            print(f"  - {m}")


def recommend_phase_b(best_configs, top_k=5):
    """Recommend configs for Phase B based on Phase A results"""
    print("\n" + "="*60)
    print(f"PHASE B RECOMMENDATIONS (Top-{top_k} configs)")
    print("="*60)

    print("\nBased on Phase A results, test these configurations on:")
    print("  - All 6 models (Base/Medium/Large × Infix/Prefix)")
    print("  - All 12 Nguyen benchmarks")
    print("  - Multiple seeds for statistical significance")

    print(f"\nTop-{top_k} configurations to test:")

    for i, r in enumerate(best_configs[:top_k], 1):
        print(f"\n{i}. Test R² = {r['test_r2']:.4f}")
        print(f"   algorithm: {r['algorithm']}")
        print(f"   reward: {r['config'].get('reward_type')}")
        print(f"   penalty: {r['config'].get('penalty_type')}")
        print(f"   temperature: {r['config'].get('temperature_schedule')}")
        print(f"   prompt: {r['config'].get('prompt_type')}")
        print(f"   noise: {r['config'].get('noise_level', 0.0)}")

    # Generate config file for Phase B
    phase_b_configs = []
    for r in best_configs[:top_k]:
        config = {
            'algorithm': r['algorithm'],
            'reward_type': r['config'].get('reward_type'),
            'penalty_type': r['config'].get('penalty_type'),
            'temperature_schedule': r['config'].get('temperature_schedule'),
            'prompt_type': r['config'].get('prompt_type'),
            'noise_level': r['config'].get('noise_level', 0.0),
            'phase_a_test_r2': r['test_r2'],
        }
        phase_b_configs.append(config)

    output_file = 'phase_b_recommended_configs.json'
    with open(output_file, 'w') as f:
        json.dump(phase_b_configs, f, indent=2)

    print(f"\nSaved recommended configs to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase A results")
    parser.add_argument('--download-from-instances', action='store_true',
                       help='Download results from AWS instances (requires running instances)')
    parser.add_argument('--use-cache', type=str,
                       help='Use cached results from JSON file')
    parser.add_argument('--save-cache', type=str, default='phase_a_results_cache.json',
                       help='Save results to cache file')
    parser.add_argument('--limit', type=int,
                       help='Limit number of results to download (for testing)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top configs to show')

    args = parser.parse_args()

    # Load results
    if args.use_cache and Path(args.use_cache).exists():
        print(f"Loading cached results from {args.use_cache}...")
        with open(args.use_cache) as f:
            results = json.load(f)
    elif args.download_from_instances:
        results = download_results_from_instances()
    else:
        results = download_results_from_hf(limit=args.limit)

    if not results:
        print("No results found!")
        return

    print(f"\nTotal results: {len(results)}")

    # Save cache
    if args.save_cache:
        with open(args.save_cache, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results cache to: {args.save_cache}")

    # Run analyses
    analyze_hyperparameter_impact(results)
    best_configs = find_best_configs(results, top_k=args.top_k)
    identify_gaps(results)
    recommend_phase_b(best_configs, top_k=5)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
