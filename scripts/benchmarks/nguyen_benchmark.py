#!/usr/bin/env python3
"""
Nguyen Symbolic Regression Benchmark

Standard benchmark suite from:
Nguyen et al. (2011) "Semantically-based crossover in genetic programming"
Genetic Programming and Evolvable Machines, 12(2), 91-119.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple, Dict
import argparse


# Nguyen Benchmark Functions
NGUYEN_BENCHMARKS: Dict[str, Dict] = {
    "nguyen_1": {
        "equation": "x**3 + x**2 + x",
        "latex": r"x^3 + x^2 + x",
        "func": lambda x: x**3 + x**2 + x,
        "n_vars": 1,
        "range": (-1, 1),
        "n_points": 20,
    },
    "nguyen_2": {
        "equation": "x**4 + x**3 + x**2 + x",
        "latex": r"x^4 + x^3 + x^2 + x",
        "func": lambda x: x**4 + x**3 + x**2 + x,
        "n_vars": 1,
        "range": (-1, 1),
        "n_points": 20,
    },
    "nguyen_3": {
        "equation": "x**5 + x**4 + x**3 + x**2 + x",
        "latex": r"x^5 + x^4 + x^3 + x^2 + x",
        "func": lambda x: x**5 + x**4 + x**3 + x**2 + x,
        "n_vars": 1,
        "range": (-1, 1),
        "n_points": 20,
    },
    "nguyen_4": {
        "equation": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
        "latex": r"x^6 + x^5 + x^4 + x^3 + x^2 + x",
        "func": lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
        "n_vars": 1,
        "range": (-1, 1),
        "n_points": 20,
    },
    "nguyen_5": {
        "equation": "sin(x**2)*cos(x) - 1",
        "latex": r"\sin(x^2) \cos(x) - 1",
        "func": lambda x: np.sin(x**2) * np.cos(x) - 1,
        "n_vars": 1,
        "range": (-1, 1),
        "n_points": 20,
    },
    "nguyen_6": {
        "equation": "sin(x) + sin(x + x**2)",
        "latex": r"\sin(x) + \sin(x + x^2)",
        "func": lambda x: np.sin(x) + np.sin(x + x**2),
        "n_vars": 1,
        "range": (-1, 1),
        "n_points": 20,
    },
    "nguyen_7": {
        "equation": "log(x + 1) + log(x**2 + 1)",
        "latex": r"\ln(x+1) + \ln(x^2+1)",
        "func": lambda x: np.log(x + 1) + np.log(x**2 + 1),
        "n_vars": 1,
        "range": (0, 2),
        "n_points": 20,
    },
    "nguyen_8": {
        "equation": "sqrt(x)",
        "latex": r"\sqrt{x}",
        "func": lambda x: np.sqrt(x),
        "n_vars": 1,
        "range": (0, 4),
        "n_points": 20,
    },
    # Two-variable benchmarks
    "nguyen_9": {
        "equation": "sin(x) + sin(y**2)",
        "latex": r"\sin(x) + \sin(y^2)",
        "func": lambda x, y: np.sin(x) + np.sin(y**2),
        "n_vars": 2,
        "range": (-1, 1),
        "n_points": 100,
    },
    "nguyen_10": {
        "equation": "2*sin(x)*cos(y)",
        "latex": r"2 \sin(x) \cos(y)",
        "func": lambda x, y: 2 * np.sin(x) * np.cos(y),
        "n_vars": 2,
        "range": (-1, 1),
        "n_points": 100,
    },
    "nguyen_11": {
        "equation": "x**y",
        "latex": r"x^y",
        "func": lambda x, y: np.power(x, y),
        "n_vars": 2,
        "range": (0, 1),
        "n_points": 100,
    },
    "nguyen_12": {
        "equation": "x**4 - x**3 + y**2/2 - y",
        "latex": r"x^4 - x^3 + \frac{y^2}{2} - y",
        "func": lambda x, y: x**4 - x**3 + y**2/2 - y,
        "n_vars": 2,
        "range": (0, 1),
        "n_points": 100,
    },
}


def generate_dataset(
    benchmark_name: str,
    n_samples: int = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate dataset for a Nguyen benchmark.

    Args:
        benchmark_name: Name of benchmark (e.g., 'nguyen_1')
        n_samples: Number of samples (default uses benchmark's n_points)
        seed: Random seed for reproducibility

    Returns:
        X: Input array of shape (n_samples, n_vars)
        y: Output array of shape (n_samples,)
        info: Benchmark metadata
    """
    if benchmark_name not in NGUYEN_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    np.random.seed(seed)

    bench = NGUYEN_BENCHMARKS[benchmark_name]
    n_vars = bench["n_vars"]
    low, high = bench["range"]
    n_points = n_samples or bench["n_points"]
    func = bench["func"]

    # Generate random inputs
    X = np.random.uniform(low, high, size=(n_points, n_vars))

    # Compute outputs
    if n_vars == 1:
        y = func(X[:, 0])
    elif n_vars == 2:
        y = func(X[:, 0], X[:, 1])
    else:
        raise ValueError(f"Unsupported n_vars: {n_vars}")

    info = {
        "name": benchmark_name,
        "equation": bench["equation"],
        "latex": bench["latex"],
        "n_vars": n_vars,
        "range": bench["range"],
        "n_samples": n_points,
    }

    return X, y, info


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    info: Dict = None
):
    """Save dataset as CSV file."""
    n_vars = X.shape[1]

    # Create DataFrame
    columns = [f"x_{i+1}" for i in range(n_vars)]
    df = pd.DataFrame(X, columns=columns)
    df["y"] = y

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Save metadata if provided
    if info:
        meta_path = output_path.with_suffix(".meta.txt")
        with open(meta_path, "w") as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")

    print(f"Saved {output_path} ({len(df)} samples)")


def generate_all_benchmarks(
    output_dir: Path,
    n_samples: int = 100,
    seed: int = 42
):
    """Generate all Nguyen benchmark datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NGUYEN SYMBOLIC REGRESSION BENCHMARKS")
    print("=" * 60)

    for name in NGUYEN_BENCHMARKS:
        bench = NGUYEN_BENCHMARKS[name]
        X, y, info = generate_dataset(name, n_samples=n_samples, seed=seed)

        output_path = output_dir / f"{name}.csv"
        save_dataset(X, y, output_path, info)

        print(f"  {name}: {bench['equation']}")

    print("=" * 60)
    print(f"Generated {len(NGUYEN_BENCHMARKS)} benchmark datasets in {output_dir}")


def list_benchmarks():
    """Print all available benchmarks."""
    print("\nNguyen Symbolic Regression Benchmarks")
    print("=" * 70)
    print(f"{'Name':<12} {'Vars':<6} {'Range':<12} {'Equation':<40}")
    print("-" * 70)

    for name, bench in NGUYEN_BENCHMARKS.items():
        range_str = f"[{bench['range'][0]}, {bench['range'][1]}]"
        print(f"{name:<12} {bench['n_vars']:<6} {range_str:<12} {bench['equation']:<40}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Nguyen symbolic regression benchmark datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/benchmarks/nguyen",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Generate specific benchmark only"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available benchmarks"
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    if args.benchmark:
        X, y, info = generate_dataset(args.benchmark, args.n_samples, args.seed)
        output_path = Path(args.output_dir) / f"{args.benchmark}.csv"
        save_dataset(X, y, output_path, info)
    else:
        generate_all_benchmarks(args.output_dir, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
