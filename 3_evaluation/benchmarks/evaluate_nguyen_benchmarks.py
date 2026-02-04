#!/usr/bin/env python3
"""
Evaluate models on Nguyen benchmarks with R² scoring.
Generates candidate expressions and calculates fit quality.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from classes.expression import Expression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_auto(model_path: str):
    """Load model with automatic base model detection"""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"No adapter_config.json in {model_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path", "gpt2")
    logger.info(f"Loading base model: {base_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer, base_model_name


def load_nguyen_benchmark(csv_path: str):
    """Load Nguyen benchmark data"""
    df = pd.read_csv(csv_path)

    # Extract X and y
    y_col = 'y'
    x_cols = [col for col in df.columns if col != y_col]

    X = df[x_cols].values
    y = df[y_col].values

    # Read metadata if available
    meta_path = csv_path.replace('.csv', '.meta.txt')
    true_formula = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                if 'formula:' in line.lower() or 'expression:' in line.lower():
                    true_formula = line.split(':', 1)[1].strip()
                    break

    return X, y, x_cols, true_formula


def create_json_prompt(x_cols, operators=None):
    """Create JSON format prompt for expression generation"""
    if operators is None:
        operators = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "abs"]

    prompt = {
        "vars": x_cols,
        "ops": operators,
        "cons": "C",
        "expr": ""
    }

    prompt_str = json.dumps(prompt, ensure_ascii=False)
    prompt_str = prompt_str.rsplit('"expr":', 1)[0] + '"expr": "'

    return prompt_str


def extract_expression_json(output: str):
    """Extract expression from JSON output"""
    import re

    # Try to find complete JSON "expr": "..."
    match = re.search(r'"expr":\s*"([^"]*)"', output)
    if match:
        return match.group(1)

    # Try to find partial JSON
    match = re.search(r'"expr":\s*"([^"]+)', output)
    if match:
        expr = match.group(1)
        expr = expr.split('"')[0].split('}')[0].strip()
        return expr

    return None


def evaluate_model_on_benchmark(model, tokenizer, X, y, x_cols, num_samples=100):
    """Evaluate model on a single benchmark"""
    device = model.device

    results = {
        "expressions": [],
        "valid_count": 0,
        "r2_scores": [],
        "best_r2": -float('inf'),
        "best_expression": None
    }

    logger.info(f"Generating {num_samples} candidate expressions...")

    for i in tqdm(range(num_samples), desc="Generating"):
        prompt = create_json_prompt(x_cols)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        expr_str = extract_expression_json(generated)

        is_valid = False
        r2 = -float('inf')
        error_msg = None

        if expr_str:
            try:
                expr = Expression(expr_str, is_prefix=False)

                # Check if expression is valid
                if expr.sympy_expression is not None:
                    # Try to evaluate on dataset
                    try:
                        if expr.is_valid_on_dataset(X):
                            is_valid = True
                            results["valid_count"] += 1

                            # Fit constants and compute R²
                            try:
                                r2 = expr.fit_constants(X, y)

                                if np.isfinite(r2):
                                    results["r2_scores"].append(r2)

                                    if r2 > results["best_r2"]:
                                        results["best_r2"] = r2
                                        results["best_expression"] = expr_str
                                else:
                                    r2 = -float('inf')
                                    error_msg = "Non-finite R²"
                            except Exception as e:
                                error_msg = f"Fit error: {str(e)[:100]}"
                        else:
                            error_msg = "Invalid on dataset"
                    except Exception as e:
                        error_msg = f"Evaluation error: {str(e)[:100]}"
            except Exception as e:
                error_msg = f"Parse error: {str(e)[:100]}"
        else:
            error_msg = "Failed to extract expression"

        results["expressions"].append({
            "index": i,
            "expression": expr_str,
            "valid": is_valid,
            "r2": float(r2) if np.isfinite(r2) else None,
            "error": error_msg
        })

    # Compute summary statistics
    valid_rate = results["valid_count"] / num_samples if num_samples > 0 else 0
    r2_scores = results["r2_scores"]

    results["summary"] = {
        "num_samples": num_samples,
        "valid_count": results["valid_count"],
        "valid_rate": valid_rate,
        "num_with_r2": len(r2_scores),
        "best_r2": float(results["best_r2"]) if np.isfinite(results["best_r2"]) else None,
        "mean_r2": float(np.mean(r2_scores)) if r2_scores else None,
        "median_r2": float(np.median(r2_scores)) if r2_scores else None,
        "std_r2": float(np.std(r2_scores)) if r2_scores else None,
        "best_expression": results["best_expression"]
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--benchmark_csv", type=str, required=True, help="Path to Nguyen benchmark CSV")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of candidate expressions to generate")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info(f"Evaluating: {os.path.basename(args.model_path)}")
    logger.info(f"Benchmark: {os.path.basename(args.benchmark_csv)}")
    logger.info("="*60)

    # Load model
    model, tokenizer, base_model_name = load_model_auto(args.model_path)

    # Load benchmark
    X, y, x_cols, true_formula = load_nguyen_benchmark(args.benchmark_csv)
    logger.info(f"Loaded benchmark: {X.shape[0]} samples, {len(x_cols)} variables")
    if true_formula:
        logger.info(f"True formula: {true_formula}")

    # Evaluate
    results = evaluate_model_on_benchmark(model, tokenizer, X, y, x_cols, args.num_samples)

    # Add metadata
    results["metadata"] = {
        "model_path": args.model_path,
        "base_model": base_model_name,
        "benchmark_csv": args.benchmark_csv,
        "true_formula": true_formula,
        "num_variables": len(x_cols),
        "num_data_points": len(y)
    }

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Valid expressions: {results['summary']['valid_count']}/{results['summary']['num_samples']} ({results['summary']['valid_rate']*100:.1f}%)")
    logger.info(f"Expressions with R²: {results['summary']['num_with_r2']}")

    if results['summary']['best_r2'] is not None:
        logger.info(f"Best R²: {results['summary']['best_r2']:.6f}")
        logger.info(f"Mean R²: {results['summary']['mean_r2']:.6f}")
        logger.info(f"Median R²: {results['summary']['median_r2']:.6f}")
        logger.info(f"Best expression: {results['summary']['best_expression']}")
    else:
        logger.info("No valid R² scores obtained")

    logger.info(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
