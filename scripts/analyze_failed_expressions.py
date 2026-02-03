#!/usr/bin/env python3
"""
Analyze failed expressions from experiment logs.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from expression import Expression


def analyze_expression(expr_str: str, X: np.ndarray, y: np.ndarray) -> dict:
    """Analyze why an expression failed."""
    result = {
        "expression": expr_str,
        "error_type": None,
        "error_message": None,
        "r2": -1.0,
        "is_valid": False,
    }

    # Replace C with 1 for testing
    test_expr = expr_str.replace('C', '1')

    try:
        # Try to parse
        expr = Expression(test_expr, is_prefix=False)

        # Try to validate on dataset
        if not expr.is_valid_on_dataset(X):
            result["error_type"] = "invalid_on_dataset"
            result["error_message"] = "Expression produces invalid values (NaN/Inf) on dataset"
            return result

        # Try to evaluate
        y_pred = expr.evaluate(X)

        if not np.all(np.isfinite(y_pred)):
            result["error_type"] = "non_finite_output"
            result["error_message"] = f"Expression produces NaN/Inf values"
            return result

        # Compute R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)

        result["r2"] = float(np.clip(r2, -1.0, 1.0))
        result["is_valid"] = True

        if r2 < -0.5:
            result["error_type"] = "very_poor_fit"
            result["error_message"] = f"R²={r2:.4f} (worse than constant predictor)"
        elif r2 < 0.0:
            result["error_type"] = "poor_fit"
            result["error_message"] = f"R²={r2:.4f} (negative R²)"

    except Exception as e:
        result["error_type"] = "parse_error"
        result["error_message"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze failed expressions")
    parser.add_argument("--log_file", type=str, required=True, help="Path to experiment log JSON")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--max_expressions", type=int, default=20, help="Max expressions to analyze")
    args = parser.parse_args()

    # Load log
    with open(args.log_file, 'r') as f:
        log_data = json.load(f)

    # Load dataset
    import pandas as pd
    df = pd.read_csv(args.dataset)
    x_cols = [c for c in df.columns if c.startswith('x_')]
    X = df[x_cols].values
    y = df['y'].values

    print(f"Dataset: {args.dataset}")
    print(f"  Samples: {len(df)}, Variables: {len(x_cols)}")
    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
    print()

    # Get expressions
    discovered = log_data.get('discovered_expressions', {})

    if not discovered:
        print("No discovered expressions found in log!")
        return

    print(f"Found {len(discovered)} expressions in log")
    print(f"Analyzing first {args.max_expressions}...")
    print()

    # Analyze
    error_counts = {}
    results = []

    for i, (expr_str, logged_r2) in enumerate(list(discovered.items())[:args.max_expressions]):
        result = analyze_expression(expr_str, X, y)
        results.append(result)

        error_type = result["error_type"] or "success"
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

        print(f"{i+1:2d}. {expr_str[:60]:<60} | {result['error_type'] or 'OK':<20} | R²={result['r2']:.4f}")
        if result["error_message"]:
            print(f"    > {result['error_message']}")

    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total analyzed: {len(results)}")
    print()
    print("Error types:")
    for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        print(f"  {error_type:<30} {count:3d} ({pct:5.1f}%)")

    # Show a few examples of each error type
    print()
    print("Examples by error type:")
    print()

    for error_type in sorted(set(r["error_type"] for r in results if r["error_type"])):
        examples = [r for r in results if r["error_type"] == error_type][:3]
        print(f"{error_type}:")
        for ex in examples:
            print(f"  - {ex['expression'][:70]}")
            print(f"    {ex['error_message']}")
        print()


if __name__ == "__main__":
    main()
