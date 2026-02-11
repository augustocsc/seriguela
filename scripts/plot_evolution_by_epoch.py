#!/usr/bin/env python3
"""
Generate scatter plots showing R² evolution across epochs
Each point = one expression generated at that epoch
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_report(json_file):
    """Load report with full history"""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_epochs_data(experiment):
    """Extract all expressions across all epochs from an experiment"""
    epochs_data = []

    # Get history from experiment
    history = experiment.get('history', [])

    for epoch_data in history:
        epoch = epoch_data.get('epoch', 0)
        expressions = epoch_data.get('expressions', [])

        for expr in expressions:
            r2 = expr.get('r2', -1.0)
            is_valid = expr.get('is_valid', False)
            epochs_data.append({
                'epoch': epoch,
                'r2': r2,
                'is_valid': is_valid,
                'expression': expr.get('expression', '')
            })

    return epochs_data

def plot_experiment_evolution(model, benchmark, algorithm, epochs_data, output_file):
    """Plot scatter of R² scores across epochs"""

    # Separate valid and invalid
    valid_epochs = [d['epoch'] for d in epochs_data if d['is_valid']]
    valid_r2 = [d['r2'] for d in epochs_data if d['is_valid']]

    invalid_epochs = [d['epoch'] for d in epochs_data if not d['is_valid']]
    invalid_r2 = [d['r2'] for d in epochs_data if not d['is_valid']]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot invalid expressions (red, small)
    if invalid_epochs:
        ax.scatter(invalid_epochs, invalid_r2,
                  c='red', alpha=0.3, s=20, label='Invalid')

    # Plot valid expressions (blue, larger)
    if valid_epochs:
        ax.scatter(valid_epochs, valid_r2,
                  c='blue', alpha=0.6, s=40, label='Valid')

    # Find best expression
    if valid_r2:
        best_idx = np.argmax(valid_r2)
        best_epoch = valid_epochs[best_idx]
        best_r2 = valid_r2[best_idx]

        # Mark best with star
        ax.scatter([best_epoch], [best_r2],
                  c='gold', marker='*', s=500,
                  edgecolors='black', linewidths=2,
                  label=f'Best (R²={best_r2:.4f})', zorder=10)

    # Style
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model} + {benchmark} + {algorithm.upper()}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    # Add horizontal line at R² = 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Set y-axis limits
    all_r2 = valid_r2 + invalid_r2
    if all_r2:
        ymin = min(all_r2) - 0.1
        ymax = max(all_r2) + 0.1
        ax.set_ylim(ymin, ymax)

    # Add statistics text
    stats_text = f'Total: {len(epochs_data)}\n'
    stats_text += f'Valid: {len(valid_r2)} ({len(valid_r2)/len(epochs_data)*100:.1f}%)\n'
    if valid_r2:
        stats_text += f'Best R²: {max(valid_r2):.4f}\n'
        stats_text += f'Avg R² (valid): {np.mean(valid_r2):.4f}'

    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    print("Loading raw_results.json...")
    raw_file = Path("evaluation_results_aws/raw_results.json")

    with open(raw_file, 'r') as f:
        experiments = json.load(f)

    print(f"Loaded {len(experiments)} experiments")

    # Create output directory
    output_dir = Path("evaluation_results_aws/plots/evolution_by_epoch")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each experiment
    experiment_counter = 0

    # We need to extract model, benchmark, algorithm from raw data
    # The raw_results.json contains full experiments with history

    # Load report.json to get the mapping
    report_file = Path("evaluation_results_aws/report.json")
    with open(report_file, 'r') as f:
        report = json.load(f)

    summary_table = report['summary_table']

    # Match experiments from raw_results with summary_table
    for idx, exp in enumerate(experiments):
        if not exp.get('success', False):
            continue

        # Get metadata from summary_table (same order)
        if idx < len(summary_table):
            meta = summary_table[idx]
            model = meta['model']
            benchmark = meta['benchmark']
            algorithm = meta['algorithm']
        else:
            print(f"Warning: No metadata for experiment {idx}")
            continue

        print(f"[{experiment_counter+1}/{len(experiments)}] Processing: {model} + {benchmark} + {algorithm}")

        # Extract epochs data
        epochs_data = extract_epochs_data(exp)

        if not epochs_data:
            print(f"  [SKIP] No data found")
            continue

        # Generate plot
        output_file = output_dir / f"{model}_{benchmark}_{algorithm}.png"
        plot_experiment_evolution(model, benchmark, algorithm, epochs_data, output_file)

        print(f"  [OK] Saved: {output_file}")
        experiment_counter += 1

    print(f"\n[DONE] Generated {experiment_counter} plots in {output_dir}")
    print(f"\nPlots saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
