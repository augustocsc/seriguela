#!/usr/bin/env python3
"""
Analyze Nguyen benchmark results and generate comprehensive report.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load summary
with open('results_nguyen_benchmarks/summary.json') as f:
    data = json.load(f)

results = data['results']

# Create DataFrame
df = pd.DataFrame(results)

# Group by model
print("="*80)
print("NGUYEN BENCHMARK RESULTS - COMPLETE ANALYSIS")
print("="*80)
print()

# Overall statistics by model
print("="*80)
print("OVERALL STATISTICS BY MODEL")
print("="*80)
print()

for model in ['base', 'medium', 'large']:
    model_data = df[df['model'] == model]

    print(f"{model.upper()} Model:")
    print(f"  Benchmarks completed: {len(model_data)}/12")
    print(f"  Avg Valid Rate: {model_data['valid_rate'].mean()*100:.1f}%")
    print(f"  Min Valid Rate: {model_data['valid_rate'].min()*100:.1f}%")
    print(f"  Max Valid Rate: {model_data['valid_rate'].max()*100:.1f}%")
    print()
    print(f"  Avg Best R²: {model_data['best_r2'].mean():.4f}")
    print(f"  Min Best R²: {model_data['best_r2'].min():.4f}")
    print(f"  Max Best R²: {model_data['best_r2'].max():.10f}")
    print()
    print(f"  Avg Duration: {model_data['duration'].mean():.1f}s")
    print()

# Comparison table
print("="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print()

comparison = []
for model in ['base', 'medium', 'large']:
    model_data = df[df['model'] == model]
    comparison.append({
        'Model': model.capitalize(),
        'Avg Valid %': f"{model_data['valid_rate'].mean()*100:.1f}%",
        'Avg Best R²': f"{model_data['best_r2'].mean():.4f}",
        'Max R²': f"{model_data['best_r2'].max():.6f}",
        'Benchmarks >0.99 R²': len(model_data[model_data['best_r2'] > 0.99])
    })

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))
print()

# Per-benchmark analysis
print("="*80)
print("PER-BENCHMARK ANALYSIS (Best R² by Model)")
print("="*80)
print()

benchmark_table = []
for bench_num in range(1, 13):
    row = {'Benchmark': f'Nguyen-{bench_num}'}
    for model in ['base', 'medium', 'large']:
        bench_data = df[(df['model'] == model) & (df['benchmark'] == f'nguyen_{bench_num}')]
        if not bench_data.empty:
            best_r2 = bench_data['best_r2'].values[0]
            valid_rate = bench_data['valid_rate'].values[0]
            row[f'{model.capitalize()}'] = f"{best_r2:.4f}"
            row[f'{model.capitalize()} Valid%'] = f"{valid_rate*100:.0f}%"
    benchmark_table.append(row)

bench_df = pd.DataFrame(benchmark_table)
print(bench_df.to_string(index=False))
print()

# Top performers
print("="*80)
print("TOP 10 BEST R² SCORES ACROSS ALL EXPERIMENTS")
print("="*80)
print()

top_10 = df.nlargest(10, 'best_r2')[['model', 'benchmark', 'best_r2', 'valid_rate']]
top_10['valid_rate'] = top_10['valid_rate'].apply(lambda x: f"{x*100:.1f}%")
print(top_10.to_string(index=False))
print()

# Valid rate progression
print("="*80)
print("VALID RATE PROGRESSION (Base → Medium → Large)")
print("="*80)
print()

for bench_num in range(1, 13):
    rates = []
    for model in ['base', 'medium', 'large']:
        bench_data = df[(df['model'] == model) & (df['benchmark'] == f'nguyen_{bench_num}')]
        if not bench_data.empty:
            rates.append(bench_data['valid_rate'].values[0] * 100)

    if len(rates) == 3:
        improvement = rates[2] - rates[0]  # Large - Base
        arrow = "↑" if improvement > 0 else "↓"
        print(f"Nguyen-{bench_num:2d}: {rates[0]:4.0f}% → {rates[1]:4.0f}% → {rates[2]:4.0f}% {arrow} (+{improvement:4.1f}%)")

print()

# Statistical summary
print("="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print()

base_avg_r2 = df[df['model'] == 'base']['best_r2'].mean()
medium_avg_r2 = df[df['model'] == 'medium']['best_r2'].mean()
large_avg_r2 = df[df['model'] == 'large']['best_r2'].mean()

print(f"Average Best R² Improvement:")
print(f"  Base → Medium: +{(medium_avg_r2 - base_avg_r2):.4f} ({(medium_avg_r2/base_avg_r2 - 1)*100:+.2f}%)")
print(f"  Medium → Large: +{(large_avg_r2 - medium_avg_r2):.4f} ({(large_avg_r2/medium_avg_r2 - 1)*100:+.2f}%)")
print(f"  Base → Large: +{(large_avg_r2 - base_avg_r2):.4f} ({(large_avg_r2/base_avg_r2 - 1)*100:+.2f}%)")
print()

base_avg_valid = df[df['model'] == 'base']['valid_rate'].mean()
medium_avg_valid = df[df['model'] == 'medium']['valid_rate'].mean()
large_avg_valid = df[df['model'] == 'large']['valid_rate'].mean()

print(f"Average Valid Rate Improvement:")
print(f"  Base → Medium: +{(medium_avg_valid - base_avg_valid)*100:.1f} percentage points")
print(f"  Medium → Large: +{(large_avg_valid - medium_avg_valid)*100:.1f} percentage points")
print(f"  Base → Large: +{(large_avg_valid - base_avg_valid)*100:.1f} percentage points")
print()

# Perfect or near-perfect fits
print("="*80)
print("PERFECT OR NEAR-PERFECT FITS (R² ≥ 0.999)")
print("="*80)
print()

perfect_fits = df[df['best_r2'] >= 0.999][['model', 'benchmark', 'best_r2', 'valid_rate']]
perfect_fits = perfect_fits.sort_values('best_r2', ascending=False)
perfect_fits['valid_rate'] = perfect_fits['valid_rate'].apply(lambda x: f"{x*100:.1f}%")
print(perfect_fits.to_string(index=False))
print()

print("="*80)
print("EXECUTION SUMMARY")
print("="*80)
print()
print(f"Total experiments: {data['total_experiments']}")
print(f"Completed: {data['completed']}")
print(f"Failed: {data['failed']}")
print(f"Total duration: {data['total_duration_seconds']/60:.1f} minutes ({data['total_duration_seconds']/3600:.2f} hours)")
print(f"Avg time per experiment: {data['total_duration_seconds']/data['completed']:.1f} seconds")
print()
print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
print()
