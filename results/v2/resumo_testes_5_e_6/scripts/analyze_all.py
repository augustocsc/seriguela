#!/usr/bin/env python3
"""Comprehensive analysis of all 36 JSON result files."""
import json
import os
import glob
from collections import defaultdict

base = r'c:\Users\madeinweb\seriguela\results\resumo_testes_5_e_6\jsons'
files = sorted(glob.glob(os.path.join(base, '*.json')))

print(f"Total files: {len(files)}\n")

# First, explore the structure of one file
with open(files[0], 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"=== Structure of {os.path.basename(files[0])} ===")
print(f"Top-level type: {type(data).__name__}")
if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
            if len(v) > 0:
                if isinstance(v[0], dict):
                    print(f"    first item keys: {list(v[0].keys())[:15]}")
                else:
                    print(f"    first item type: {type(v[0]).__name__}, value: {v[0]}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())[:10]}")
        else:
            print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")

print("\n" + "="*80)

# Now extract key metrics from all files
results = []
for fpath in files:
    fname = os.path.basename(fpath)
    parts = fname.replace('.json', '').replace('aggregate_', '').split('_')
    
    # Parse name: e.g. bon_grpo_nguyen_1_seed123
    algo_type = parts[0]  # bon or pure
    algo = parts[1]       # grpo or ppo
    problem = parts[2] + '_' + parts[3]  # nguyen_1, nguyen_5, nguyen_9
    seed = parts[4].replace('seed', '')
    
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Try to extract metrics
    info = {
        'algo_type': algo_type,
        'algo': algo,
        'full_algo': f"{algo_type}_{algo}",
        'problem': problem,
        'seed': seed,
        'file': fname,
        'file_size_kb': os.path.getsize(fpath) / 1024,
    }
    
    # Extract training data depending on structure
    if isinstance(data, dict):
        # Check common key patterns
        for key in ['history', 'training_history', 'steps', 'log', 'results']:
            if key in data and isinstance(data[key], list):
                history = data[key]
                info['n_steps'] = len(history)
                if len(history) > 0:
                    last = history[-1]
                    if isinstance(last, dict):
                        # Try extracting R2 and reward
                        for r2_key in ['best_r2', 'r2', 'eval_r2', 'test_r2', 'best_test_r2']:
                            if r2_key in last:
                                info['final_r2'] = last[r2_key]
                                break
                        for rew_key in ['mean_reward', 'reward', 'avg_reward']:
                            if rew_key in last:
                                info['final_reward'] = last[rew_key]
                                break
                    # Best R2 across all steps
                    best_r2 = None
                    for entry in history:
                        if isinstance(entry, dict):
                            for r2_key in ['best_r2', 'r2', 'eval_r2', 'test_r2']:
                                if r2_key in entry:
                                    val = entry[r2_key]
                                    if val is not None and (best_r2 is None or val > best_r2):
                                        best_r2 = val
                                    break
                    info['best_r2'] = best_r2
                break
        
        # Check if data itself looks like a flat results dict
        for r2_key in ['best_r2', 'r2', 'final_r2', 'best_test_r2']:
            if r2_key in data:
                info['best_r2'] = data[r2_key]
                break
        
        # Check nested config
        if 'config' in data:
            config = data['config']
            if isinstance(config, dict):
                info['batch_size'] = config.get('batch_size')
                info['max_steps'] = config.get('max_steps')
                info['temperature'] = config.get('temperature')
                info['reward_fn'] = config.get('reward_function', config.get('reward'))
    
    results.append(info)

# Print summary table
print("\n=== ALL RESULTS ===")
print(f"{'Algorithm':<15} {'Problem':<12} {'Seed':<6} {'Steps':<7} {'Best R2':<10} {'Final R2':<10} {'Size(KB)':<10}")
print("-" * 80)
for r in results:
    print(f"{r['full_algo']:<15} {r['problem']:<12} {r['seed']:<6} {str(r.get('n_steps', '?')):<7} {str(r.get('best_r2', '?')):<10} {str(r.get('final_r2', '?')):<10} {r['file_size_kb']:<10.1f}")

# Aggregate by algorithm and problem
print("\n\n=== AVERAGES BY ALGORITHM x PROBLEM ===")
agg = defaultdict(list)
for r in results:
    key = (r['full_algo'], r['problem'])
    if r.get('best_r2') is not None:
        agg[key].append(r['best_r2'])

print(f"{'Algorithm':<15} {'Problem':<12} {'Mean Best R2':<12} {'Min':<10} {'Max':<10} {'N'}")
print("-" * 65)
for key in sorted(agg.keys()):
    vals = agg[key]
    mean_val = sum(vals) / len(vals)
    print(f"{key[0]:<15} {key[1]:<12} {mean_val:<12.4f} {min(vals):<10.4f} {max(vals):<10.4f} {len(vals)}")

# BoN vs Pure comparison
print("\n\n=== BoN vs PURE COMPARISON (per algorithm, per problem) ===")
for problem in ['nguyen_1', 'nguyen_5', 'nguyen_9']:
    print(f"\n--- {problem} ---")
    for algo in ['grpo', 'ppo']:
        bon_key = (f'bon_{algo}', problem)
        pure_key = (f'pure_{algo}', problem)
        bon_vals = agg.get(bon_key, [])
        pure_vals = agg.get(pure_key, [])
        if bon_vals and pure_vals:
            bon_mean = sum(bon_vals)/len(bon_vals)
            pure_mean = sum(pure_vals)/len(pure_vals)
            diff = bon_mean - pure_mean
            winner = "BoN" if diff > 0 else "Pure"
            print(f"  {algo.upper()}: BoN={bon_mean:.4f} vs Pure={pure_mean:.4f}  (delta={diff:+.4f}) -> {winner} wins")
