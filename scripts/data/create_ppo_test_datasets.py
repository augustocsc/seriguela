#!/usr/bin/env python3
"""
Create simple test datasets for PPO symbolic regression experiments.
No constants (C) - just simple expressions to verify PPO works.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def create_dataset(formula_func, formula_name, n_vars, n_samples=500,
                   x_range=(-2, 2), output_dir="./data/ppo_test"):
    """
    Create a synthetic regression dataset.

    Args:
        formula_func: Function that takes X array and returns y
        formula_name: Name for the dataset (used as filename)
        n_vars: Number of input variables
        n_samples: Number of data points
        x_range: Range for random X values
        output_dir: Directory to save CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random input data
    np.random.seed(42)  # For reproducibility
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, n_vars))

    # Compute target
    y = formula_func(X)

    # Create DataFrame
    columns = [f"x_{i+1}" for i in range(n_vars)] + ["y"]
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    output_file = output_dir / f"{formula_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Created: {output_file} ({n_samples} samples, {n_vars} vars)")
    print(f"  Formula: {formula_name}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")

    return output_file


def main():
    print("=" * 60)
    print("Creating PPO Test Datasets (No Constants)")
    print("=" * 60)

    # ========================================
    # EASY: Simple expressions with 2 variables
    # ========================================
    print("\n--- EASY DATASETS (2 variables) ---")

    # x_1 + x_2
    create_dataset(
        formula_func=lambda X: X[:, 0] + X[:, 1],
        formula_name="add_x1_x2",
        n_vars=2
    )

    # x_1 * x_2
    create_dataset(
        formula_func=lambda X: X[:, 0] * X[:, 1],
        formula_name="mul_x1_x2",
        n_vars=2
    )

    # x_1 - x_2
    create_dataset(
        formula_func=lambda X: X[:, 0] - X[:, 1],
        formula_name="sub_x1_x2",
        n_vars=2
    )

    # ========================================
    # MEDIUM: Unary functions
    # ========================================
    print("\n--- MEDIUM DATASETS (unary functions) ---")

    # sin(x_1)
    create_dataset(
        formula_func=lambda X: np.sin(X[:, 0]),
        formula_name="sin_x1",
        n_vars=1
    )

    # cos(x_1)
    create_dataset(
        formula_func=lambda X: np.cos(X[:, 0]),
        formula_name="cos_x1",
        n_vars=1
    )

    # x_1 * x_1 (quadratic)
    create_dataset(
        formula_func=lambda X: X[:, 0] * X[:, 0],
        formula_name="square_x1",
        n_vars=1
    )

    # ========================================
    # HARD: Composed expressions
    # ========================================
    print("\n--- HARD DATASETS (composed expressions) ---")

    # sin(x_1) + x_2
    create_dataset(
        formula_func=lambda X: np.sin(X[:, 0]) + X[:, 1],
        formula_name="sin_x1_plus_x2",
        n_vars=2
    )

    # x_1 * sin(x_2)
    create_dataset(
        formula_func=lambda X: X[:, 0] * np.sin(X[:, 1]),
        formula_name="x1_mul_sin_x2",
        n_vars=2
    )

    # sin(x_1 + x_2)
    create_dataset(
        formula_func=lambda X: np.sin(X[:, 0] + X[:, 1]),
        formula_name="sin_x1_plus_x2_composed",
        n_vars=2
    )

    # x_1 * x_2 + x_1
    create_dataset(
        formula_func=lambda X: X[:, 0] * X[:, 1] + X[:, 0],
        formula_name="x1_mul_x2_plus_x1",
        n_vars=2
    )

    print("\n" + "=" * 60)
    print("Done! Created 10 test datasets in ./data/ppo_test/")
    print("=" * 60)


if __name__ == "__main__":
    main()
