#!/usr/bin/env python3
"""
Run REINFORCE on Nguyen benchmarks and generate report.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Nguyen benchmark info
NGUYEN_BENCHMARKS = {
    "nguyen_1": "x³ + x² + x",
    "nguyen_2": "x⁴ + x³ + x² + x",
    "nguyen_3": "x⁵ + x⁴ + x³ + x² + x",
    "nguyen_4": "x⁶ + x⁵ + x⁴ + x³ + x² + x",
    "nguyen_5": "sin(x²)·cos(x) - 1",
    "nguyen_6": "sin(x) + sin(x + x²)",
    "nguyen_7": "ln(x+1) + ln(x²+1)",
    "nguyen_8": "√x",
}

def run_benchmark(
    benchmark_name: str,
    model_path: str,
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    grad_accum: int = 4,
) -> dict:
    """Run REINFORCE on a single benchmark."""
    dataset_path = f"{data_dir}/{benchmark_name}.csv"

    cmd = [
        "python", "scripts/reinforce_improved.py",
        "--model_path", model_path,
        "--dataset", dataset_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
    ]

    print(f"\n{'='*60}")
    print(f"Running {benchmark_name}: {NGUYEN_BENCHMARKS.get(benchmark_name, 'Unknown')}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        output = result.stdout + result.stderr

        # Parse results from output
        best_r2 = None
        best_expr = None
        epochs_used = None

        for line in output.split('\n'):
            if 'Best R^2:' in line:
                try:
                    best_r2 = float(line.split('Best R^2:')[1].strip())
                except:
                    pass
            if 'Best expression:' in line:
                best_expr = line.split('Best expression:')[1].strip()
            if 'Target R^2 0.99 reached at epoch' in line:
                try:
                    epochs_used = int(line.split('epoch')[1].strip().rstrip('!'))
                except:
                    pass
            if 'No improvement for' in line and epochs_used is None:
                # Early stopped without reaching target
                for prev_line in output.split('\n'):
                    if 'Epoch' in prev_line and '|' in prev_line:
                        try:
                            epochs_used = int(prev_line.split('Epoch')[1].split('|')[0].strip())
                        except:
                            pass

        return {
            "benchmark": benchmark_name,
            "equation": NGUYEN_BENCHMARKS.get(benchmark_name, "Unknown"),
            "best_r2": best_r2,
            "best_expression": best_expr,
            "epochs": epochs_used,
            "success": best_r2 is not None and best_r2 >= 0.99,
        }

    except subprocess.TimeoutExpired:
        return {
            "benchmark": benchmark_name,
            "equation": NGUYEN_BENCHMARKS.get(benchmark_name, "Unknown"),
            "best_r2": None,
            "best_expression": None,
            "epochs": None,
            "success": False,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "benchmark": benchmark_name,
            "equation": NGUYEN_BENCHMARKS.get(benchmark_name, "Unknown"),
            "best_r2": None,
            "best_expression": None,
            "epochs": None,
            "success": False,
            "error": str(e),
        }


def generate_report(results: list, output_path: str):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Count successes
    successes = sum(1 for r in results if r.get("success", False))
    total = len(results)

    report = f"""# Nguyen Benchmark Results

**Date:** {timestamp}
**Model:** augustocsc/Se124M_700K_infix_v3_json
**Algorithm:** Improved REINFORCE

## Summary

- **Passed (R² ≥ 0.99):** {successes}/{total}
- **Success Rate:** {100*successes/total:.1f}%

## Results

| Benchmark | Target Equation | Best R² | Found Expression | Epochs | Status |
|-----------|-----------------|---------|------------------|--------|--------|
"""

    for r in results:
        status = "✅" if r.get("success", False) else "❌"
        r2 = f"{r['best_r2']:.4f}" if r.get("best_r2") is not None else "N/A"
        expr = r.get("best_expression", "N/A")
        if expr and len(expr) > 30:
            expr = expr[:27] + "..."
        epochs = r.get("epochs", "N/A")

        report += f"| {r['benchmark']} | {r['equation']} | {r2} | `{expr}` | {epochs} | {status} |\n"

    report += f"""
## Analysis

### Successful Recoveries
"""

    for r in results:
        if r.get("success", False):
            report += f"- **{r['benchmark']}**: Found `{r['best_expression']}` in {r['epochs']} epochs\n"

    report += """
### Notes

- The model was trained on expressions with sin, cos, +, -, * operators
- Polynomial-only benchmarks (Nguyen 1-4, 8) are harder since model prefers trigonometric expressions
- Benchmarks with sin/cos (Nguyen 5-6) align better with training distribution
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Nguyen benchmarks")
    parser.add_argument("--model_path", default="augustocsc/Se124M_700K_infix_v3_json")
    parser.add_argument("--data_dir", default="./data/benchmarks/nguyen")
    parser.add_argument("--output", default="./output/nguyen_benchmark_report.md")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--benchmarks", nargs="+", default=None,
                       help="Specific benchmarks to run (e.g., nguyen_1 nguyen_5)")
    args = parser.parse_args()

    benchmarks_to_run = args.benchmarks or list(NGUYEN_BENCHMARKS.keys())

    results = []
    for bench in benchmarks_to_run:
        if bench in NGUYEN_BENCHMARKS:
            result = run_benchmark(
                bench,
                args.model_path,
                args.data_dir,
                epochs=args.epochs,
            )
            results.append(result)

            # Print summary
            status = "✅ PASSED" if result.get("success") else "❌ FAILED"
            r2 = result.get("best_r2", 0)
            print(f"\n{bench}: {status} (R² = {r2:.4f})")

    # Generate report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, args.output)
    print(report)

    # Save JSON results
    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
