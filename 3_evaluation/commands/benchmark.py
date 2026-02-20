"""
Benchmark evaluation command.

Evaluates model on symbolic regression benchmarks (Nguyen, Feynman, etc.):
- Generates candidate expressions
- Fits constants to data
- Calculates R² scores
- Tracks best expression found
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Add paths for imports
_eval_dir = Path(__file__).parent.parent
_project_dir = _eval_dir.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from core.model_loader import ModelLoader
from core.generator import ExpressionGenerator, GenerationConfig, PromptConfig
from core.extractor import ExpressionExtractor
from core.storage import ResultStorage
from classes.expression import Expression

logger = logging.getLogger(__name__)


# Nguyen benchmark definitions
NGUYEN_BENCHMARKS = {
    "nguyen_1": {"formula": "x**3 + x**2 + x", "vars": ["x_1"], "range": (-1, 1)},
    "nguyen_2": {"formula": "x**4 + x**3 + x**2 + x", "vars": ["x_1"], "range": (-1, 1)},
    "nguyen_3": {"formula": "x**5 + x**4 + x**3 + x**2 + x", "vars": ["x_1"], "range": (-1, 1)},
    "nguyen_4": {"formula": "x**6 + x**5 + x**4 + x**3 + x**2 + x", "vars": ["x_1"], "range": (-1, 1)},
    "nguyen_5": {"formula": "sin(x**2)*cos(x) - 1", "vars": ["x_1"], "range": (-1, 1)},
    "nguyen_6": {"formula": "sin(x) + sin(x + x**2)", "vars": ["x_1"], "range": (-1, 1)},
    "nguyen_7": {"formula": "log(x + 1) + log(x**2 + 1)", "vars": ["x_1"], "range": (0, 2)},
    "nguyen_8": {"formula": "sqrt(x)", "vars": ["x_1"], "range": (0, 4)},
    "nguyen_9": {"formula": "sin(x) + sin(y**2)", "vars": ["x_1", "x_2"], "range": (-1, 1)},
    "nguyen_10": {"formula": "2*sin(x)*cos(y)", "vars": ["x_1", "x_2"], "range": (-1, 1)},
    "nguyen_11": {"formula": "x**y", "vars": ["x_1", "x_2"], "range": (0, 1)},
    "nguyen_12": {"formula": "x**4 - x**3 + y**2/2 - y", "vars": ["x_1", "x_2"], "range": (-1, 1)},
}


@dataclass
class BenchmarkResult:
    """Result for a single expression on a benchmark."""
    idx: int
    expression: Optional[str]
    valid: bool
    r2: Optional[float]
    constants: Optional[List[float]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "expression": self.expression,
            "valid": self.valid,
            "r2": self.r2,
            "constants": self.constants,
            "error": self.error,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for benchmark evaluation."""
    benchmark_name: str
    true_formula: Optional[str]
    num_samples: int
    valid_count: int
    valid_rate: float
    num_with_r2: int
    best_r2: Optional[float]
    mean_r2: Optional[float]
    median_r2: Optional[float]
    std_r2: Optional[float]
    best_expression: Optional[str]
    best_constants: Optional[List[float]]
    r2_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "true_formula": self.true_formula,
            "num_samples": self.num_samples,
            "valid_count": self.valid_count,
            "valid_rate": self.valid_rate,
            "num_with_r2": self.num_with_r2,
            "best_r2": self.best_r2,
            "mean_r2": self.mean_r2,
            "median_r2": self.median_r2,
            "std_r2": self.std_r2,
            "best_expression": self.best_expression,
            "best_constants": self.best_constants,
        }

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"True formula: {self.true_formula}",
            f"Valid: {self.valid_count}/{self.num_samples} ({self.valid_rate:.1%})",
            f"With R²: {self.num_with_r2}",
        ]
        if self.best_r2 is not None:
            lines.extend([
                f"Best R²: {self.best_r2:.6f}",
                f"Mean R²: {self.mean_r2:.6f}" if self.mean_r2 else "Mean R²: N/A",
                f"Best expression: {self.best_expression}",
            ])
        return "\n".join(lines)


def generate_benchmark_data(benchmark_name: str, num_points: int = 100) -> tuple:
    """Generate synthetic data for a Nguyen benchmark."""
    if benchmark_name not in NGUYEN_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    info = NGUYEN_BENCHMARKS[benchmark_name]
    vars_list = info["vars"]
    formula = info["formula"]
    low, high = info["range"]

    num_vars = len(vars_list)
    X = np.random.uniform(low, high, (num_points, num_vars))

    # Evaluate true formula
    local_vars = {"x": X[:, 0] if num_vars >= 1 else None}
    if num_vars >= 2:
        local_vars["y"] = X[:, 1]

    local_vars.update({
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs,
    })

    y = eval(formula, {"__builtins__": {}}, local_vars)

    return X, y, vars_list, formula


def load_benchmark_csv(csv_path: str) -> tuple:
    """Load benchmark data from CSV file."""
    df = pd.read_csv(csv_path)

    y_col = 'y'
    x_cols = [col for col in df.columns if col != y_col]

    X = df[x_cols].values
    y = df[y_col].values

    # Try to read true formula from metadata
    meta_path = csv_path.replace('.csv', '.meta.txt')
    true_formula = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                if 'formula:' in line.lower() or 'expression:' in line.lower():
                    true_formula = line.split(':', 1)[1].strip()
                    break

    return X, y, x_cols, true_formula


def evaluate_expression_on_data(expr_str: str, X: np.ndarray, y: np.ndarray, is_prefix: bool = False) -> tuple:
    """
    Evaluate an expression on benchmark data.

    Returns:
        Tuple of (is_valid, r2_score, constants, error_message)
    """
    try:
        expr = Expression(expr_str, is_prefix=is_prefix)

        if expr.sympy_expression is None:
            return False, None, None, "Failed to parse expression"

        # Check if valid on dataset
        if not expr.is_valid_on_dataset(X):
            return False, None, None, "Expression invalid on dataset"

        # Fit constants and get R²
        r2 = expr.fit_constants(X, y)

        if not np.isfinite(r2):
            return True, None, expr.best_constants, "Non-finite R²"

        return True, float(r2), expr.best_constants, None

    except Exception as e:
        return False, None, None, str(e)[:100]


def execute_benchmark(args: argparse.Namespace):
    """
    Execute benchmark evaluation.

    Args:
        args: Parsed command line arguments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    benchmark_name = args.benchmark
    num_samples = args.num_samples

    print(f"\n{'='*60}")
    print("Seriguela Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}\n")

    # Load or generate benchmark data
    if args.csv:
        print(f"Loading benchmark from: {args.csv}")
        X, y, x_cols, true_formula = load_benchmark_csv(args.csv)
    else:
        print(f"Generating synthetic data for: {benchmark_name}")
        X, y, x_cols, true_formula = generate_benchmark_data(benchmark_name, num_points=100)

    print(f"Data: {X.shape[0]} points, {len(x_cols)} variables")
    print(f"True formula: {true_formula}\n")

    # Initialize storage
    output_dir = getattr(args, "output_dir", "results/benchmark")
    storage = ResultStorage(base_dir=output_dir)

    # Create run config
    config = {
        "type": "benchmark",
        "model": {"path": args.model},
        "benchmark": {
            "name": benchmark_name,
            "true_formula": true_formula,
            "variables": x_cols,
            "num_points": len(y),
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": 100,
        },
        "evaluation": {
            "num_samples": num_samples,
        },
    }

    run_id = storage.create_run(config)
    print(f"Run ID: {run_id}")
    print(f"Output: {storage.get_run_dir(run_id)}\n")

    # Load model
    print("Loading model...")
    start_time = time.time()

    loader = ModelLoader()
    model, tokenizer, base_model = loader.load(args.model)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s\n")

    # Detect notation from model name
    is_prefix = "prefix" in args.model.lower()

    # Setup generator
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=100,
    )
    generator = ExpressionGenerator(model, tokenizer, gen_config)
    extractor = ExpressionExtractor()

    # Build prompt with benchmark variables
    prompt_config = PromptConfig(
        vars=x_cols,
        ops=["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "**"],
        cons="C",
        format="prefix" if is_prefix else "infix",
    )
    prompt = generator.build_prompt(prompt_config)

    print(f"Notation: {'prefix' if is_prefix else 'infix'}")
    print(f"Variables: {x_cols}")
    print("Generating and evaluating expressions...\n")

    # Generate and evaluate
    results = []
    r2_scores = []
    best_r2 = -float('inf')
    best_expression = None
    best_constants = None
    valid_count = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(range(num_samples), desc="Evaluating")
    except ImportError:
        iterator = range(num_samples)

    gen_start = time.time()

    for i in iterator:
        # Generate
        outputs = generator.generate(prompt)
        output = outputs[0] if outputs else ""

        # Extract expression
        extraction = extractor.extract(output, format_hint="json")
        expr_str = extraction.expression

        # Evaluate on benchmark data
        if expr_str:
            is_valid, r2, constants, error = evaluate_expression_on_data(
                expr_str, X, y, is_prefix=is_prefix
            )
        else:
            is_valid, r2, constants, error = False, None, None, "No expression extracted"

        if is_valid:
            valid_count += 1

        if r2 is not None and np.isfinite(r2):
            r2_scores.append(r2)
            if r2 > best_r2:
                best_r2 = r2
                best_expression = expr_str
                best_constants = constants

        result = BenchmarkResult(
            idx=i,
            expression=expr_str,
            valid=is_valid,
            r2=r2,
            constants=constants,
            error=error,
        )
        results.append(result)

        # Save incrementally
        storage.save_sample(run_id, type("SampleResult", (), {"to_dict": lambda s=result: s.to_dict()})())

    gen_time = time.time() - gen_start

    # Calculate metrics
    metrics = BenchmarkMetrics(
        benchmark_name=benchmark_name,
        true_formula=true_formula,
        num_samples=num_samples,
        valid_count=valid_count,
        valid_rate=valid_count / num_samples if num_samples > 0 else 0,
        num_with_r2=len(r2_scores),
        best_r2=best_r2 if np.isfinite(best_r2) else None,
        mean_r2=float(np.mean(r2_scores)) if r2_scores else None,
        median_r2=float(np.median(r2_scores)) if r2_scores else None,
        std_r2=float(np.std(r2_scores)) if r2_scores else None,
        best_expression=best_expression,
        best_constants=best_constants,
        r2_scores=r2_scores,
    )

    # Save metrics
    metrics_path = storage.get_run_dir(run_id) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Valid expressions: {valid_count}/{num_samples} ({metrics.valid_rate:.1%})")
    print(f"Expressions with R²: {len(r2_scores)}")

    if metrics.best_r2 is not None:
        print(f"\nBest R²: {metrics.best_r2:.6f}")
        print(f"Mean R²: {metrics.mean_r2:.6f}" if metrics.mean_r2 else "")
        print(f"Median R²: {metrics.median_r2:.6f}" if metrics.median_r2 else "")
        print(f"\nBest expression: {best_expression}")
        if best_constants:
            print(f"Best constants: {best_constants}")
    else:
        print("\nNo valid R² scores obtained")

    print(f"\nTrue formula: {true_formula}")
    print(f"\nGeneration time: {gen_time:.1f}s ({gen_time/num_samples:.2f}s/sample)")
    print(f"{'='*60}")
    print(f"\nResults saved to: {storage.get_run_dir(run_id)}")
    print(f"Run ID: {run_id}")

    return run_id


def add_benchmark_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the benchmark command."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (HuggingFace repo or local path)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="nguyen_5",
        help=f"Benchmark name: {', '.join(NGUYEN_BENCHMARKS.keys())} (default: nguyen_5)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to custom benchmark CSV file (overrides --benchmark)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of candidate expressions to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark",
        help="Output directory for results (default: results/benchmark)",
    )


def list_benchmarks():
    """List available benchmarks."""
    print("\nAvailable Nguyen Benchmarks:")
    print("-" * 60)
    for name, info in NGUYEN_BENCHMARKS.items():
        print(f"  {name:12} | {info['formula']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    add_benchmark_arguments(parser)
    args = parser.parse_args()
    execute_benchmark(args)
