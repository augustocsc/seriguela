#!/usr/bin/env python3
"""
Aggregate and analyze Nguyen benchmark suite results
Part of Model Scaling Experiment (Feb 2025)

Usage:
    python scripts/aggregate_nguyen_results.py --input_dir nguyen_suite_results
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(input_dir: Path) -> pd.DataFrame:
    """Load all experiment results from JSON files."""
    results = []

    models = ['base', 'medium', 'large']
    benchmarks = range(1, 13)  # Nguyen 1-12
    algorithms = ['supervised', 'reinforce', 'grpo', 'ppo']

    for model in models:
        for bench in benchmarks:
            for algo in algorithms:
                result_file = input_dir / f"{model}_nguyen{bench}_{algo}.json"

                if not result_file.exists():
                    print(f"Warning: Missing {result_file}")
                    continue

                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)

                    # Extract key metrics
                    result = {
                        'model': model,
                        'model_size': {'base': '124M', 'medium': '355M', 'large': '774M'}[model],
                        'benchmark': f'nguyen_{bench}',
                        'algorithm': algo,
                        'valid_rate': data.get('valid_expression_rate', 0.0),
                        'best_r2': data.get('best_r2', -1.0),
                        'mean_r2': data.get('mean_r2', -1.0),
                        'power_ops_rate': data.get('power_operations_usage', 0.0),
                        'nested_trig_rate': data.get('nested_trig_functions', 0.0),
                        'avg_depth': data.get('average_depth', 0.0),
                        'constraint_adherence': data.get('constraint_adherence_rate', 0.0),
                        'diversity_rate': data.get('diversity_rate', 0.0),
                    }

                    results.append(result)

                except Exception as e:
                    print(f"Error loading {result_file}: {e}")

    df = pd.DataFrame(results)
    print(f"\nLoaded {len(df)} experiment results")
    return df


def create_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Create summary tables for each algorithm."""

    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]

        # Pivot table: benchmarks × models
        pivot = algo_df.pivot_table(
            index='benchmark',
            columns='model',
            values='best_r2',
            aggfunc='mean'
        )

        # Save to CSV
        csv_file = output_dir / f"summary_{algo}_r2.csv"
        pivot.to_csv(csv_file)
        print(f"Saved: {csv_file}")

        # Valid rate table
        valid_pivot = algo_df.pivot_table(
            index='benchmark',
            columns='model',
            values='valid_rate',
            aggfunc='mean'
        )
        valid_csv = output_dir / f"summary_{algo}_valid_rate.csv"
        valid_pivot.to_csv(valid_csv)
        print(f"Saved: {valid_csv}")


def create_aggregate_statistics(df: pd.DataFrame, output_dir: Path):
    """Create overall statistics by model and algorithm."""

    stats = df.groupby(['model', 'algorithm']).agg({
        'best_r2': ['mean', 'std', 'min', 'max'],
        'valid_rate': ['mean', 'std'],
        'power_ops_rate': ['mean', 'std'],
        'nested_trig_rate': ['mean', 'std'],
        'avg_depth': ['mean', 'std'],
        'constraint_adherence': ['mean', 'std'],
        'diversity_rate': ['mean', 'std'],
    }).round(4)

    csv_file = output_dir / "aggregate_statistics.csv"
    stats.to_csv(csv_file)
    print(f"Saved: {csv_file}")

    return stats


def plot_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Create heatmaps for visual comparison."""

    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]

        # R² heatmap
        pivot = algo_df.pivot_table(
            index='benchmark',
            columns='model',
            values='best_r2',
            aggfunc='mean'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=0, vmin=-1, vmax=1, cbar_kws={'label': 'R²'})
        plt.title(f'Best R² - {algo.upper()} Algorithm\nNguyen Benchmarks by Model Size')
        plt.xlabel('Model Size')
        plt.ylabel('Benchmark')
        plt.tight_layout()

        png_file = output_dir / f"heatmap_{algo}_r2.png"
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {png_file}")

        # Valid rate heatmap
        valid_pivot = algo_df.pivot_table(
            index='benchmark',
            columns='model',
            values='valid_rate',
            aggfunc='mean'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(valid_pivot, annot=True, fmt='.1%', cmap='YlGnBu',
                    vmin=0, vmax=1, cbar_kws={'label': 'Valid Expression Rate'})
        plt.title(f'Valid Expression Rate - {algo.upper()} Algorithm\nNguyen Benchmarks by Model Size')
        plt.xlabel('Model Size')
        plt.ylabel('Benchmark')
        plt.tight_layout()

        png_file = output_dir / f"heatmap_{algo}_valid_rate.png"
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {png_file}")


def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Create bar charts comparing models across metrics."""

    # Average metrics by model (across all benchmarks and algorithms)
    model_avg = df.groupby('model').agg({
        'best_r2': 'mean',
        'valid_rate': 'mean',
        'power_ops_rate': 'mean',
        'avg_depth': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Best R²
    axes[0, 0].bar(model_avg['model'], model_avg['best_r2'], color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 0].set_title('Average Best R² by Model Size', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Valid Rate
    axes[0, 1].bar(model_avg['model'], model_avg['valid_rate']*100, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 1].set_title('Average Valid Expression Rate by Model Size', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Valid Rate (%)')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Power Operations
    axes[1, 0].bar(model_avg['model'], model_avg['power_ops_rate']*100, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1, 0].set_title('Average Power Operations Usage by Model Size', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Usage (%)')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Average Depth
    axes[1, 1].bar(model_avg['model'], model_avg['avg_depth'], color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1, 1].set_title('Average Expression Depth by Model Size', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Depth')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.suptitle('Model Scaling Comparison\n(Averaged across all Nguyen benchmarks and algorithms)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    png_file = output_dir / "model_comparison_overview.png"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_file}")


def plot_algorithm_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare algorithms across models."""

    algo_model = df.groupby(['model', 'algorithm']).agg({
        'best_r2': 'mean',
        'valid_rate': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² by algorithm and model
    pivot_r2 = algo_model.pivot(index='model', columns='algorithm', values='best_r2')
    pivot_r2.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Average Best R² by Algorithm and Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('R²')
    axes[0].set_xlabel('Model Size')
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].legend(title='Algorithm')
    axes[0].grid(axis='y', alpha=0.3)

    # Valid rate by algorithm and model
    pivot_valid = algo_model.pivot(index='model', columns='algorithm', values='valid_rate')
    (pivot_valid * 100).plot(kind='bar', ax=axes[1], rot=0)
    axes[1].set_title('Average Valid Rate by Algorithm and Model', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Valid Rate (%)')
    axes[1].set_xlabel('Model Size')
    axes[1].legend(title='Algorithm')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    png_file = output_dir / "algorithm_comparison.png"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_file}")


def generate_markdown_report(df: pd.DataFrame, stats: pd.DataFrame, output_dir: Path):
    """Generate a comprehensive markdown report."""

    report = []
    report.append("# Nguyen Benchmark Suite Results")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Total Experiments:** {len(df)}")
    report.append(f"\n**Models:** Base (124M), Medium (355M), Large (774M)")
    report.append(f"\n**Benchmarks:** Nguyen 1-12")
    report.append(f"\n**Algorithms:** Supervised, REINFORCE, GRPO, PPO")

    report.append("\n## Overall Statistics\n")
    report.append("### Average Best R² by Model\n")

    model_r2 = df.groupby('model')['best_r2'].mean()
    report.append("| Model | Average Best R² |")
    report.append("|-------|----------------|")
    for model, r2 in model_r2.items():
        report.append(f"| {model} | {r2:.4f} |")

    report.append("\n### Average Valid Expression Rate by Model\n")
    model_valid = df.groupby('model')['valid_rate'].mean()
    report.append("| Model | Valid Rate |")
    report.append("|-------|-----------|")
    for model, rate in model_valid.items():
        report.append(f"| {model} | {rate*100:.1f}% |")

    report.append("\n### Complexity Metrics by Model\n")
    complexity = df.groupby('model').agg({
        'power_ops_rate': 'mean',
        'nested_trig_rate': 'mean',
        'avg_depth': 'mean',
    })

    report.append("| Model | Power Ops | Nested Trig | Avg Depth |")
    report.append("|-------|-----------|-------------|-----------|")
    for model in complexity.index:
        report.append(f"| {model} | {complexity.loc[model, 'power_ops_rate']*100:.1f}% | "
                     f"{complexity.loc[model, 'nested_trig_rate']*100:.1f}% | "
                     f"{complexity.loc[model, 'avg_depth']:.2f} |")

    report.append("\n## Algorithm Performance\n")

    for algo in ['supervised', 'reinforce', 'grpo', 'ppo']:
        algo_df = df[df['algorithm'] == algo]
        report.append(f"\n### {algo.upper()}\n")

        algo_model = algo_df.groupby('model').agg({
            'best_r2': ['mean', 'std'],
            'valid_rate': 'mean',
        }).round(4)

        report.append("| Model | Best R² (mean ± std) | Valid Rate |")
        report.append("|-------|---------------------|-----------|")
        for model in ['base', 'medium', 'large']:
            if model in algo_model.index:
                r2_mean = algo_model.loc[model, ('best_r2', 'mean')]
                r2_std = algo_model.loc[model, ('best_r2', 'std')]
                valid = algo_model.loc[model, ('valid_rate', 'mean')]
                report.append(f"| {model} | {r2_mean:.4f} ± {r2_std:.4f} | {valid*100:.1f}% |")

    report.append("\n## Key Findings\n")

    # Compare base vs large
    base_r2 = df[df['model'] == 'base']['best_r2'].mean()
    large_r2 = df[df['model'] == 'large']['best_r2'].mean()
    improvement = ((large_r2 - base_r2) / abs(base_r2)) * 100 if base_r2 != 0 else 0

    report.append(f"\n1. **R² Improvement (Base → Large):** {improvement:+.1f}%")

    # Valid rate improvement
    base_valid = df[df['model'] == 'base']['valid_rate'].mean()
    large_valid = df[df['model'] == 'large']['valid_rate'].mean()
    valid_improvement = (large_valid - base_valid) * 100

    report.append(f"2. **Valid Rate Improvement (Base → Large):** {valid_improvement:+.1f} percentage points")

    # Complexity improvement
    base_depth = df[df['model'] == 'base']['avg_depth'].mean()
    large_depth = df[df['model'] == 'large']['avg_depth'].mean()
    depth_improvement = ((large_depth - base_depth) / base_depth) * 100

    report.append(f"3. **Expression Depth Increase (Base → Large):** {depth_improvement:+.1f}%")

    # Best algorithm
    algo_performance = df.groupby('algorithm')['best_r2'].mean().sort_values(ascending=False)
    best_algo = algo_performance.index[0]

    report.append(f"4. **Best Algorithm Overall:** {best_algo.upper()} (R² = {algo_performance.iloc[0]:.4f})")

    report.append("\n## Visualizations\n")
    report.append("\n- `model_comparison_overview.png`: Overall model comparison")
    report.append("- `algorithm_comparison.png`: Algorithm performance comparison")
    report.append("- `heatmap_*_r2.png`: R² heatmaps by algorithm")
    report.append("- `heatmap_*_valid_rate.png`: Valid rate heatmaps by algorithm")

    report.append("\n## Data Files\n")
    report.append("\n- `aggregate_statistics.csv`: Detailed statistics")
    report.append("- `summary_*_r2.csv`: R² tables by algorithm")
    report.append("- `summary_*_valid_rate.csv`: Valid rate tables by algorithm")
    report.append("- `full_results.csv`: Complete raw data")

    # Write report
    report_file = output_dir / "RESULTS_REPORT.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate Nguyen benchmark results')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing experiment JSON files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    print("="*50)
    print("Nguyen Results Aggregation")
    print("="*50)

    # Load results
    print("\n[1/7] Loading results...")
    df = load_results(input_dir)

    if df.empty:
        print("Error: No results loaded")
        return

    # Save complete dataset
    full_csv = input_dir / "full_results.csv"
    df.to_csv(full_csv, index=False)
    print(f"Saved: {full_csv}")

    # Create summary tables
    print("\n[2/7] Creating summary tables...")
    create_summary_tables(df, input_dir)

    # Aggregate statistics
    print("\n[3/7] Computing aggregate statistics...")
    stats = create_aggregate_statistics(df, input_dir)

    # Create heatmaps
    print("\n[4/7] Generating heatmaps...")
    plot_heatmaps(df, input_dir)

    # Model comparison plots
    print("\n[5/7] Creating model comparison plots...")
    plot_model_comparison(df, input_dir)

    # Algorithm comparison plots
    print("\n[6/7] Creating algorithm comparison plots...")
    plot_algorithm_comparison(df, input_dir)

    # Generate markdown report
    print("\n[7/7] Generating markdown report...")
    generate_markdown_report(df, stats, input_dir)

    print("\n" + "="*50)
    print("✓ Aggregation Complete!")
    print("="*50)
    print(f"\nResults saved to: {input_dir}/")
    print(f"Main report: {input_dir}/RESULTS_REPORT.md")
    print()


if __name__ == "__main__":
    main()
