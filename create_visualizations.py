#!/usr/bin/env python3
"""
Create visualizations for Model Scaling Study.
Generates publication-ready charts and tables.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create output directory
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("CREATING VISUALIZATIONS FOR MODEL SCALING STUDY")
print("="*80)
print()

# Load data
print("Loading data...")

# Quality results
quality_data = {
    'Base': {'valid_rate': 0.994, 'diversity': 0.978, 'unique': 489, 'samples': 500},
    'Medium': {'valid_rate': 0.992, 'diversity': 0.988, 'unique': 494, 'samples': 500},
    'Large': {'valid_rate': 1.000, 'diversity': 0.986, 'unique': 493, 'samples': 500}
}

# Nguyen benchmark summary
with open('results_nguyen_benchmarks/summary.json') as f:
    nguyen_data = json.load(f)

# Extract Nguyen stats by model
nguyen_stats = {}
for model in ['base', 'medium', 'large']:
    model_results = [r for r in nguyen_data['results'] if r['model'] == model]
    nguyen_stats[model.capitalize()] = {
        'avg_valid_rate': np.mean([r['valid_rate'] for r in model_results]),
        'avg_best_r2': np.mean([r['best_r2'] for r in model_results]),
        'max_r2': max([r['best_r2'] for r in model_results]),
        'benchmarks_gt_099': sum([1 for r in model_results if r['best_r2'] > 0.99])
    }

print("Data loaded successfully!")
print()

# ============================================================================
# Figure 1: Valid Rate Comparison (Quality + Benchmarks)
# ============================================================================
print("Creating Figure 1: Valid Rate Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

models = ['Base', 'Medium', 'Large']
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Quality valid rates
quality_valid = [quality_data[m]['valid_rate'] * 100 for m in models]
bars1 = ax1.bar(models, quality_valid, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Valid Expression Rate (%)', fontsize=14, fontweight='bold')
ax1.set_title('Quality Evaluation\n(500 samples per model)', fontsize=16, fontweight='bold')
ax1.set_ylim([95, 101])
ax1.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Perfect (100%)')
ax1.legend()

# Add value labels on bars
for bar, val in zip(bars1, quality_valid):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Benchmark valid rates
benchmark_valid = [nguyen_stats[m]['avg_valid_rate'] * 100 for m in models]
bars2 = ax2.bar(models, benchmark_valid, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Valid Expression Rate (%)', fontsize=14, fontweight='bold')
ax2.set_title('Nguyen Benchmarks\n(36 experiments, 3,600 expressions)', fontsize=16, fontweight='bold')
ax2.set_ylim([0, 100])

# Add value labels on bars
for bar, val in zip(bars2, benchmark_valid):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig1_valid_rate_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'fig1_valid_rate_comparison.png'}")
plt.close()

# ============================================================================
# Figure 2: R² Performance on Nguyen Benchmarks
# ============================================================================
print("Creating Figure 2: R² Performance...")

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.25

avg_r2 = [nguyen_stats[m]['avg_best_r2'] for m in models]
max_r2 = [nguyen_stats[m]['max_r2'] for m in models]

bars1 = ax.bar(x - width/2, avg_r2, width, label='Average Best R²',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, max_r2, width, label='Maximum R²',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax.set_title('Symbolic Regression Performance (Nguyen Benchmarks)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=12)
ax.set_ylim([0.85, 1.05])
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Fit')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig2_r2_performance.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'fig2_r2_performance.png'}")
plt.close()

# ============================================================================
# Figure 3: Per-Benchmark Heatmap
# ============================================================================
print("Creating Figure 3: Per-Benchmark Heatmap...")

# Extract per-benchmark R² scores
benchmark_matrix = []
for bench in range(1, 13):
    row = []
    for model in ['base', 'medium', 'large']:
        result = [r for r in nguyen_data['results']
                 if r['model'] == model and r['benchmark'] == f'nguyen_{bench}']
        if result:
            row.append(result[0]['best_r2'])
        else:
            row.append(0)
    benchmark_matrix.append(row)

benchmark_matrix = np.array(benchmark_matrix)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(benchmark_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

# Set ticks
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(12))
ax.set_xticklabels(['Base\n(124M)', 'Medium\n(355M)', 'Large\n(774M)'], fontsize=12)
ax.set_yticklabels([f'Nguyen-{i+1}' for i in range(12)], fontsize=11)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('R² Score', rotation=270, labelpad=20, fontsize=14, fontweight='bold')

# Add text annotations
for i in range(12):
    for j in range(3):
        text = ax.text(j, i, f'{benchmark_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('R² Scores by Model and Benchmark', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'fig3_benchmark_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'fig3_benchmark_heatmap.png'}")
plt.close()

# ============================================================================
# Figure 4: Scaling Progression (Valid Rate + R²)
# ============================================================================
print("Creating Figure 4: Scaling Progression...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

params = [124, 355, 774]  # Millions

# Valid rate progression (benchmarks)
ax1.plot(params, benchmark_valid, 'o-', color='#3498db', linewidth=3,
         markersize=12, label='Nguyen Valid Rate', markeredgecolor='black', markeredgewidth=2)
ax1.set_xlabel('Model Size (Million Parameters)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Valid Expression Rate (%)', fontsize=14, fontweight='bold')
ax1.set_title('Valid Rate vs Model Size', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)

# Add value labels
for x, y in zip(params, benchmark_valid):
    ax1.text(x, y + 2, f'{y:.1f}%', ha='center', fontsize=11, fontweight='bold')

# R² progression
ax2.plot(params, avg_r2, 'o-', color='#e74c3c', linewidth=3,
         markersize=12, label='Average Best R²', markeredgecolor='black', markeredgewidth=2)
ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Fit')
ax2.set_xlabel('Model Size (Million Parameters)', fontsize=14, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax2.set_title('R² vs Model Size', fontsize=16, fontweight='bold')
ax2.set_ylim([0.9, 1.02])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)

# Add value labels
for x, y in zip(params, avg_r2):
    ax2.text(x, y + 0.005, f'{y:.4f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig4_scaling_progression.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'fig4_scaling_progression.png'}")
plt.close()

print()
print("="*80)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*80)
print()
print(f"Output directory: {output_dir.absolute()}")
print()
print("Generated files:")
print("  1. fig1_valid_rate_comparison.png - Quality vs Benchmark valid rates")
print("  2. fig2_r2_performance.png - R² scores comparison")
print("  3. fig3_benchmark_heatmap.png - Per-benchmark R² heatmap")
print("  4. fig4_scaling_progression.png - Scaling laws visualization")
print()
print("These figures are publication-ready (300 DPI, high resolution)")
print()
