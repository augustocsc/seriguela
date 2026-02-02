#!/usr/bin/env python3
"""
Best-of-N Sampling Experiment for Symbolic Regression

Instead of PPO (which has API compatibility issues with TRL 0.16+),
this script tests if the base model can find correct expressions
through random sampling. If the model generates the correct expression
even occasionally, PPO should be able to learn to find it consistently.

This is a diagnostic experiment to understand model capabilities.
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from expression import Expression
from dataset import RegressionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BestOfNSampler:
    """Generate N expressions and find the best one for a given dataset."""

    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path

        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self._load_model()

    def _load_model(self):
        """Load the JSON format model with LoRA adapters."""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer from trained model (has special tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer loaded with vocab size: {len(self.tokenizer)}")

        # Load base GPT-2
        base_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
        )

        # Resize embeddings to match tokenizer (handles special tokens)
        if len(self.tokenizer) != base_model.config.vocab_size:
            logger.info(f"Resizing embeddings: {base_model.config.vocab_size} -> {len(self.tokenizer)}")
            base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        try:
            model_with_lora = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = model_with_lora.merge_and_unload()
            logger.info("LoRA adapter loaded and merged")
        except Exception as e:
            logger.warning(f"Could not load as PEFT model: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def build_prompt(self, n_vars: int) -> str:
        """Build JSON format prompt matching training data."""
        vars_list = [f"x_{i+1}" for i in range(n_vars)]
        ops_list = ["+", "-", "*", "sin", "cos"]

        prompt = json.dumps({
            "vars": vars_list,
            "ops": ops_list,
            "cons": None,
            "expr": ""
        })[:-3]  # Remove trailing '"}'

        return prompt

    def extract_expression(self, generated_text: str) -> str:
        """Extract expression from JSON format output."""
        try:
            # Find the expression field
            if '"expr": "' in generated_text:
                expr_start = generated_text.index('"expr": "') + len('"expr": "')
            elif '"expr":"' in generated_text:
                expr_start = generated_text.index('"expr":"') + len('"expr":"')
            else:
                return generated_text.split('"expr"')[-1].strip(' ":}')

            # Find the end - look for closing "} pattern
            remaining = generated_text[expr_start:]

            # Try to find proper JSON end first - look for "} which closes the expr field
            if '"}' in remaining:
                expr_end = remaining.index('"}')
                extracted = remaining[:expr_end].strip()
                return extracted

            # Fallback: find first quote that ends the expression
            if '"' in remaining:
                expr_end = remaining.index('"')
                return remaining[:expr_end].strip()

            return remaining.strip()
        except (ValueError, IndexError):
            pass
        # Last resort fallback
        fallback = generated_text.split('"expr"')[-1].strip(' ":}')
        # Clean up any overflow
        if '"}' in fallback:
            fallback = fallback[:fallback.index('"}')]
        return fallback

    def compute_r2(self, expression_str: str, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score for an expression."""
        if not expression_str or expression_str.isspace():
            return -np.inf

        # Replace constant placeholder C with 1
        if 'C' in expression_str:
            expression_str = expression_str.replace('C', '1')

        try:
            expr = Expression(expression_str, is_prefix=False)

            if not expr.is_valid_on_dataset(X):
                return -np.inf

            y_pred = expr.evaluate(X)

            if not np.all(np.isfinite(y_pred)):
                return -np.inf

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                return 0.0

            return 1 - (ss_res / ss_tot)
        except Exception:
            return -np.inf

    def sample_expressions(self, n_vars: int, n_samples: int = 100,
                           temperature: float = 0.7) -> list:
        """Generate N expression samples."""
        prompt = self.build_prompt(n_vars)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        expressions = []

        debug_count = 0
        for _ in tqdm(range(n_samples), desc="Sampling expressions"):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            expr_str = self.extract_expression(text)

            # Debug: print first 5 extractions
            if debug_count < 5:
                logger.info(f"DEBUG [{debug_count}] raw text (last 80 chars): ...{text[-80:]}")
                logger.info(f"DEBUG [{debug_count}] extracted: '{expr_str}'")
                debug_count += 1

            expressions.append(expr_str)

        return expressions

    def find_best_expression(self, X: np.ndarray, y: np.ndarray,
                             n_samples: int = 500, temperature: float = 0.7):
        """Sample N expressions and find the best one for the dataset."""
        n_vars = X.shape[1]

        logger.info(f"Sampling {n_samples} expressions for {n_vars}-variable dataset...")
        expressions = self.sample_expressions(n_vars, n_samples, temperature)

        # Compute R² for each
        results = []
        unique_expressions = set()

        for expr_str in tqdm(expressions, desc="Computing R² scores"):
            if expr_str in unique_expressions:
                continue
            unique_expressions.add(expr_str)

            r2 = self.compute_r2(expr_str, X, y)
            results.append({
                "expression": expr_str,
                "r2": float(r2) if np.isfinite(r2) else None,
                "is_valid": bool(np.isfinite(r2) and r2 > -1),
            })

        # Sort by R²
        results.sort(key=lambda x: x["r2"] if x["r2"] is not None else -np.inf, reverse=True)

        # Statistics
        valid_count = sum(1 for r in results if r["is_valid"])
        valid_r2s = [r["r2"] for r in results if r["r2"] is not None and r["r2"] > -1]

        return {
            "n_samples": n_samples,
            "unique_expressions": len(unique_expressions),
            "valid_count": valid_count,
            "valid_rate": valid_count / len(unique_expressions) if unique_expressions else 0,
            "best_r2": results[0]["r2"] if results and results[0]["r2"] else None,
            "best_expression": results[0]["expression"] if results else None,
            "mean_r2": float(np.mean(valid_r2s)) if valid_r2s else None,
            "median_r2": float(np.median(valid_r2s)) if valid_r2s else None,
            "top_10": results[:10],
        }


def run_experiment(model_path: str, datasets_dir: str, n_samples: int = 500,
                   output_dir: str = "./output/best_of_n"):
    """Run Best-of-N experiment on multiple datasets."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test datasets
    test_datasets = {
        "add_x1_x2": {"formula": "x_1 + x_2", "difficulty": "easy"},
        "mul_x1_x2": {"formula": "x_1 * x_2", "difficulty": "easy"},
        "sub_x1_x2": {"formula": "x_1 - x_2", "difficulty": "easy"},
        "sin_x1": {"formula": "sin(x_1)", "difficulty": "medium"},
        "cos_x1": {"formula": "cos(x_1)", "difficulty": "medium"},
        "square_x1": {"formula": "x_1 * x_1", "difficulty": "medium"},
        "sin_x1_plus_x2": {"formula": "sin(x_1) + x_2", "difficulty": "hard"},
        "x1_mul_sin_x2": {"formula": "x_1 * sin(x_2)", "difficulty": "hard"},
    }

    # Initialize sampler
    sampler = BestOfNSampler(model_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": timestamp,
        "model_path": model_path,
        "n_samples": n_samples,
        "datasets": {},
    }

    print("\n" + "=" * 70)
    print("BEST-OF-N SAMPLING EXPERIMENT")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Samples per dataset: {n_samples}")
    print("=" * 70)

    for dataset_name, info in test_datasets.items():
        dataset_path = Path(datasets_dir) / f"{dataset_name}.csv"

        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Ground truth: {info['formula']}")
        print(f"Difficulty: {info['difficulty']}")
        print(f"{'='*70}")

        # Load dataset
        reg = RegressionDataset(str(dataset_path.parent), dataset_path.name)
        X, y = reg.get_numpy()

        # Run Best-of-N
        result = sampler.find_best_expression(X, y, n_samples)
        result["ground_truth"] = info["formula"]
        result["difficulty"] = info["difficulty"]

        results["datasets"][dataset_name] = result

        # Print results
        print(f"\nResults:")
        print(f"  Valid expressions: {result['valid_count']}/{result['unique_expressions']} ({result['valid_rate']:.1%})")
        print(f"  Best R²: {result['best_r2']:.4f}" if result['best_r2'] else "  Best R²: N/A")
        print(f"  Best expression: {result['best_expression']}")

        if result['best_r2'] and result['best_r2'] > 0.99:
            print(f"  ✅ FOUND NEAR-PERFECT MATCH!")
        elif result['best_r2'] and result['best_r2'] > 0.9:
            print(f"  ⚠️  Found good match (R² > 0.9)")
        else:
            print(f"  ❌ No good match found")

        print("\n  Top 5 expressions:")
        for i, expr in enumerate(result['top_10'][:5]):
            r2_str = f"{expr['r2']:.4f}" if expr['r2'] else "N/A"
            print(f"    {i+1}. {expr['expression'][:40]:<40} R²={r2_str}")

    # Save results
    results_file = output_dir / f"best_of_n_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Summary table
    print(f"\n{'Dataset':<25} {'Difficulty':<10} {'Best R²':<10} {'Found?':<10}")
    print("-" * 60)

    success_count = 0
    for name, res in results["datasets"].items():
        r2 = res["best_r2"]
        r2_str = f"{r2:.4f}" if r2 else "N/A"
        found = "✅" if r2 and r2 > 0.99 else ("⚠️" if r2 and r2 > 0.9 else "❌")
        if r2 and r2 > 0.99:
            success_count += 1
        print(f"{name:<25} {res['difficulty']:<10} {r2_str:<10} {found:<10}")

    print("-" * 60)
    print(f"Success rate (R² > 0.99): {success_count}/{len(results['datasets'])}")
    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Best-of-N Sampling Experiment")
    parser.add_argument("--model_path", type=str, default="./output/exp_a_json",
                        help="Path to trained model")
    parser.add_argument("--datasets_dir", type=str, default="./data/ppo_test",
                        help="Directory containing test datasets")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of samples per dataset")
    parser.add_argument("--output_dir", type=str, default="./output/best_of_n",
                        help="Output directory for results")

    args = parser.parse_args()

    run_experiment(
        model_path=args.model_path,
        datasets_dir=args.datasets_dir,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
