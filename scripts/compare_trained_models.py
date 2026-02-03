#!/usr/bin/env python3
"""
Compare trained models (base vs medium) on expression generation complexity.
Runs REINFORCE for a few epochs on Nguyen-5 to see which model explores better.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from scripts.debug_reinforce import DebugREINFORCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, required=True, help="Path to trained base model")
    parser.add_argument("--model_medium", type=str, required=True, help="Path to trained medium model")
    parser.add_argument("--dataset", type=str, default="data/benchmarks/nguyen/nguyen_5.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="model_comparison.json")
    args = parser.parse_args()

    # Load dataset
    import pandas as pd
    df = pd.read_csv(args.dataset)
    x_cols = [c for c in df.columns if c.startswith('x_')]
    X = df[x_cols].values
    y = df['y'].values

    print("="*80)
    print("COMPARING TRAINED MODELS")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"  Samples: {len(df)}, Variables: {len(x_cols)}")
    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
    print()

    results = {}

    for model_name, model_path in [("base", args.model_base), ("medium", args.model_medium)]:
        print("="*80)
        print(f"Testing {model_name.upper()} model: {model_path}")
        print("="*80)

        # Run REINFORCE
        reinforce = DebugREINFORCE(model_path, X, y)
        reinforce.run(epochs=args.epochs)

        # Analyze expressions
        expressions = reinforce.all_expressions
        valid = [e for e in expressions if e["is_valid"]]

        # Count complexity features
        has_power = sum(1 for e in valid if '**' in e['expression'] or 'pow(' in e['expression'])
        has_nested_trig = sum(1 for e in valid
                             if any(nested in e['expression']
                                    for nested in ['sin(sin', 'sin(cos', 'cos(sin', 'cos(cos']))

        depths = []
        for e in valid:
            depth = max(e['expression'].count('('), 1)
            depths.append(depth)

        best_r2 = max((e['r2'] for e in expressions), default=-1.0)

        results[model_name] = {
            "model_path": model_path,
            "total_expressions": len(expressions),
            "valid_count": len(valid),
            "valid_pct": 100 * len(valid) / len(expressions) if expressions else 0,
            "has_power": has_power,
            "has_power_pct": 100 * has_power / len(valid) if valid else 0,
            "has_nested_trig": has_nested_trig,
            "has_nested_trig_pct": 100 * has_nested_trig / len(valid) if valid else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "max_depth": max(depths) if depths else 0,
            "best_r2": float(best_r2),
        }

        print()
        print(f"Results for {model_name}:")
        print(f"  Valid: {len(valid)}/{len(expressions)} ({results[model_name]['valid_pct']:.1f}%)")
        print(f"  With power: {has_power} ({results[model_name]['has_power_pct']:.1f}%)")
        print(f"  With nested trig: {has_nested_trig} ({results[model_name]['has_nested_trig_pct']:.1f}%)")
        print(f"  Avg depth: {results[model_name]['avg_depth']:.2f}")
        print(f"  Best R2: {results[model_name]['best_r2']:.4f}")
        print()

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Base':>15} {'Medium':>15} {'Improvement':>15}")
    print("-"*80)

    metrics = [
        ("Valid %", "valid_pct", "%"),
        ("Power %", "has_power_pct", "%"),
        ("Nested Trig %", "has_nested_trig_pct", "%"),
        ("Avg Depth", "avg_depth", ""),
        ("Best R2", "best_r2", ""),
    ]

    for label, key, unit in metrics:
        base_val = results["base"][key]
        medium_val = results["medium"][key]

        if base_val != 0:
            improvement = ((medium_val - base_val) / abs(base_val)) * 100
            print(f"{label:<25} {base_val:>14.2f}{unit} {medium_val:>14.2f}{unit} {improvement:>+14.1f}%")
        else:
            print(f"{label:<25} {base_val:>14.2f}{unit} {medium_val:>14.2f}{unit} {'N/A':>15}")

    print()
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
