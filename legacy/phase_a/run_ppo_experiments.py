#!/usr/bin/env python3
"""
Run PPO experiments on multiple test datasets and compare results.

This script:
1. Creates test datasets (if they don't exist)
2. Runs PPO on each dataset
3. Compares PPO vs baseline (no training)
4. Generates a summary report
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

# Test datasets with their ground truth formulas
TEST_DATASETS = {
    # EASY
    "add_x1_x2": {
        "formula": "x_1 + x_2",
        "n_vars": 2,
        "difficulty": "easy",
    },
    "mul_x1_x2": {
        "formula": "x_1 * x_2",
        "n_vars": 2,
        "difficulty": "easy",
    },
    "sub_x1_x2": {
        "formula": "x_1 - x_2",
        "n_vars": 2,
        "difficulty": "easy",
    },
    # MEDIUM
    "sin_x1": {
        "formula": "sin(x_1)",
        "n_vars": 1,
        "difficulty": "medium",
    },
    "cos_x1": {
        "formula": "cos(x_1)",
        "n_vars": 1,
        "difficulty": "medium",
    },
    "square_x1": {
        "formula": "x_1 * x_1",
        "n_vars": 1,
        "difficulty": "medium",
    },
    # HARD
    "sin_x1_plus_x2": {
        "formula": "sin(x_1) + x_2",
        "n_vars": 2,
        "difficulty": "hard",
    },
    "x1_mul_sin_x2": {
        "formula": "x_1 * sin(x_2)",
        "n_vars": 2,
        "difficulty": "hard",
    },
}


def create_datasets():
    """Create test datasets if they don't exist."""
    script_path = PROJECT_ROOT / "scripts" / "data" / "create_ppo_test_datasets.py"
    data_dir = PROJECT_ROOT / "data" / "ppo_test"

    # Check if datasets exist
    if data_dir.exists() and len(list(data_dir.glob("*.csv"))) >= len(TEST_DATASETS):
        print("Test datasets already exist.")
        return

    print("Creating test datasets...")
    subprocess.run([sys.executable, str(script_path)], check=True)


def run_baseline_evaluation(model_path: str, dataset_path: str, n_samples: int = 100):
    """
    Evaluate baseline: generate expressions without PPO training.
    Returns the best R² found among random samples.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from expression import Expression
    from dataset import RegressionDataset
    import torch

    print(f"  Running baseline evaluation ({n_samples} samples)...")

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    except:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset_path = Path(dataset_path)
    reg = RegressionDataset(str(dataset_path.parent), dataset_path.name)
    X, y = reg.get_numpy()
    n_vars = X.shape[1]

    # Build prompt
    vars_list = [f"x_{i+1}" for i in range(n_vars)]
    prompt = json.dumps({
        "vars": vars_list,
        "ops": ["+", "-", "*", "sin", "cos"],
        "cons": None,
        "expr": ""
    })[:-3]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate samples
    best_r2 = -np.inf
    best_expr = None
    valid_count = 0
    all_r2 = []

    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract expression
        try:
            if '"expr": "' in text:
                expr_start = text.index('"expr": "') + len('"expr": "')
                expr_end = text.index('"', expr_start)
                expr_str = text[expr_start:expr_end].strip()
            else:
                expr_str = text.split('"expr"')[-1].strip(' ":}')
        except:
            continue

        # Skip expressions with constants
        if 'C' in expr_str or not expr_str:
            continue

        # Compute R²
        try:
            expr = Expression(expr_str, is_prefix=False)
            if not expr.is_valid_on_dataset(X):
                continue

            valid_count += 1
            y_pred = expr.evaluate(X)

            if not np.all(np.isfinite(y_pred)):
                continue

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            all_r2.append(r2)

            if r2 > best_r2:
                best_r2 = r2
                best_expr = expr_str
        except:
            continue

    return {
        "best_r2": float(best_r2) if best_r2 > -np.inf else None,
        "best_expr": best_expr,
        "valid_rate": valid_count / n_samples,
        "mean_r2": float(np.mean(all_r2)) if all_r2 else None,
        "n_samples": n_samples,
    }


def run_ppo_experiment(model_path: str, dataset_path: str, output_dir: str,
                       batch_size: int = 32, epochs: int = 5):
    """Run PPO experiment on a single dataset."""
    from ppo_experiment import PPOSymbolicRegression

    print(f"  Running PPO experiment...")

    experiment = PPOSymbolicRegression(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=1e-5,
        max_retries=5,
    )

    results = experiment.run(n_epochs=epochs, early_stop_r2=0.95)

    return {
        "best_r2": results["best_r2"],
        "best_expr": results["best_expression"],
        "epochs_run": len(results["epochs"]),
        "final_valid_rate": results["epochs"][-1]["valid_rate"] if results["epochs"] else 0,
    }


def run_all_experiments(model_path: str, batch_size: int = 32, epochs: int = 5,
                        baseline_samples: int = 100, skip_baseline: bool = False):
    """Run experiments on all test datasets."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "output" / "ppo_experiments" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "timestamp": timestamp,
        "config": {
            "model_path": model_path,
            "batch_size": batch_size,
            "epochs": epochs,
            "baseline_samples": baseline_samples,
        },
        "experiments": {},
    }

    print("=" * 70)
    print("PPO SYMBOLIC REGRESSION EXPERIMENTS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Output: {results_dir}")
    print(f"Datasets: {len(TEST_DATASETS)}")
    print("=" * 70)

    for dataset_name, dataset_info in TEST_DATASETS.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"Ground truth: {dataset_info['formula']}")
        print(f"Difficulty: {dataset_info['difficulty']}")
        print(f"{'='*70}")

        dataset_path = PROJECT_ROOT / "data" / "ppo_test" / f"{dataset_name}.csv"

        if not dataset_path.exists():
            print(f"  ERROR: Dataset not found: {dataset_path}")
            continue

        exp_output_dir = results_dir / dataset_name

        # Run baseline
        if not skip_baseline:
            baseline_results = run_baseline_evaluation(
                model_path, str(dataset_path), baseline_samples
            )
            print(f"  Baseline: R²={baseline_results['best_r2']:.4f}" if baseline_results['best_r2'] else "  Baseline: No valid expressions")
        else:
            baseline_results = None

        # Run PPO
        ppo_results = run_ppo_experiment(
            model_path, str(dataset_path), str(exp_output_dir),
            batch_size, epochs
        )
        print(f"  PPO: R²={ppo_results['best_r2']:.4f}" if ppo_results['best_r2'] else "  PPO: No valid expressions")

        # Compare
        all_results["experiments"][dataset_name] = {
            "ground_truth": dataset_info["formula"],
            "difficulty": dataset_info["difficulty"],
            "baseline": baseline_results,
            "ppo": ppo_results,
        }

        if baseline_results and baseline_results["best_r2"] and ppo_results["best_r2"]:
            improvement = ppo_results["best_r2"] - baseline_results["best_r2"]
            print(f"  Improvement: {improvement:+.4f}")
            all_results["experiments"][dataset_name]["improvement"] = improvement

    # Generate summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_table = []
    for name, exp in all_results["experiments"].items():
        baseline_r2 = exp["baseline"]["best_r2"] if exp.get("baseline") and exp["baseline"].get("best_r2") else "N/A"
        ppo_r2 = exp["ppo"]["best_r2"] if exp["ppo"]["best_r2"] else "N/A"
        improvement = exp.get("improvement", "N/A")

        if isinstance(baseline_r2, float):
            baseline_r2 = f"{baseline_r2:.4f}"
        if isinstance(ppo_r2, float):
            ppo_r2 = f"{ppo_r2:.4f}"
        if isinstance(improvement, float):
            improvement = f"{improvement:+.4f}"

        summary_table.append({
            "Dataset": name,
            "Difficulty": exp["difficulty"],
            "Ground Truth": exp["ground_truth"],
            "Baseline R²": baseline_r2,
            "PPO R²": ppo_r2,
            "Improvement": improvement,
            "PPO Expression": exp["ppo"].get("best_expr", "N/A"),
        })

    # Print table
    print(f"\n{'Dataset':<25} {'Diff':<8} {'Baseline':<10} {'PPO':<10} {'Improve':<10}")
    print("-" * 70)
    for row in summary_table:
        print(f"{row['Dataset']:<25} {row['Difficulty']:<8} {row['Baseline R²']:<10} {row['PPO R²']:<10} {row['Improvement']:<10}")

    # Save results
    results_file = results_dir / "summary.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run PPO experiments on test datasets")
    parser.add_argument("--model_path", type=str, default="./output/exp_a_json",
                        help="Path to trained JSON format model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for PPO")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of PPO epochs per dataset")
    parser.add_argument("--baseline_samples", type=int, default=100,
                        help="Number of samples for baseline evaluation")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline evaluation")
    parser.add_argument("--create_datasets_only", action="store_true",
                        help="Only create datasets, don't run experiments")

    args = parser.parse_args()

    # Ensure datasets exist
    create_datasets()

    if args.create_datasets_only:
        print("Datasets created. Exiting.")
        return

    # Run experiments
    run_all_experiments(
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        baseline_samples=args.baseline_samples,
        skip_baseline=args.skip_baseline,
    )


if __name__ == "__main__":
    main()
