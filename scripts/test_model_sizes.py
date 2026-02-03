#!/usr/bin/env python3
"""
Test different model sizes on expression generation.
Compare GPT-2 (124M), GPT-2-medium (355M), GPT-2-large (774M).
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

from transformers import AutoTokenizer, AutoModelForCausalLM
from expression import Expression


def generate_expressions(model_name: str, num_samples: int = 20, device: str = None):
    """Generate expressions with a given model."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Build prompt (JSON format)
    vars_list = ["x_1"]
    ops_list = ["+", "-", "*", "/", "sin", "cos", "sqrt", "log", "exp", "pow"]
    prompt = json.dumps({"vars": vars_list, "ops": ops_list, "cons": "C", "expr": ""})[:-2]

    expressions = []
    valid_count = 0
    has_power = 0
    has_nested_trig = 0
    depths = []

    print(f"Generating {num_samples} expressions...")

    for i in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract expression
        expr_str = ""
        if '"expr": "' in text:
            start = text.index('"expr": "') + len('"expr": "')
            remaining = text[start:]
            for terminator in ['"}', '"']:
                if terminator in remaining:
                    expr_str = remaining[:remaining.index(terminator)].strip()
                    break

        if not expr_str:
            continue

        # Validate
        test_expr = expr_str.replace('C', '1')
        is_valid = False

        try:
            expr = Expression(test_expr, is_prefix=False)
            # Simple validation - just check if it parses
            is_valid = True
        except:
            pass

        # Count features
        if is_valid:
            valid_count += 1

            if '**' in expr_str or 'pow(' in expr_str:
                has_power += 1

            if any(nested in expr_str for nested in ['sin(sin', 'sin(cos', 'cos(sin', 'cos(cos']):
                has_nested_trig += 1

            depth = max(expr_str.count('('), 1)
            depths.append(depth)

        expressions.append({
            "expression": expr_str,
            "is_valid": is_valid,
        })

    # Stats
    stats = {
        "model_name": model_name,
        "total": len(expressions),
        "valid": valid_count,
        "valid_pct": 100 * valid_count / len(expressions) if expressions else 0,
        "has_power": has_power,
        "has_power_pct": 100 * has_power / valid_count if valid_count > 0 else 0,
        "has_nested_trig": has_nested_trig,
        "has_nested_trig_pct": 100 * has_nested_trig / valid_count if valid_count > 0 else 0,
        "avg_depth": sum(depths) / len(depths) if depths else 0,
        "max_depth": max(depths) if depths else 0,
    }

    return expressions, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt2", "gpt2-medium"],
                        help="Models to test")
    parser.add_argument("--num_samples", type=int, default=20, help="Samples per model")
    parser.add_argument("--output_file", type=str, default="model_size_comparison.json")
    args = parser.parse_args()

    results = {}

    for model_name in args.models:
        print()
        print("="*80)
        print(f"Testing {model_name}")
        print("="*80)

        expressions, stats = generate_expressions(model_name, args.num_samples)

        results[model_name] = {
            "stats": stats,
            "expressions": expressions,
        }

        print()
        print(f"Results for {model_name}:")
        print(f"  Valid: {stats['valid']}/{stats['total']} ({stats['valid_pct']:.1f}%)")
        print(f"  With power: {stats['has_power']} ({stats['has_power_pct']:.1f}%)")
        print(f"  With nested trig: {stats['has_nested_trig']} ({stats['has_nested_trig_pct']:.1f}%)")
        print(f"  Avg depth: {stats['avg_depth']:.2f}")
        print(f"  Max depth: {stats['max_depth']}")

        # Show examples
        print()
        print("Sample expressions:")
        valid_exprs = [e for e in expressions if e["is_valid"]][:5]
        for i, e in enumerate(valid_exprs, 1):
            print(f"  {i}. {e['expression'][:70]}")

    # Save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved results to {args.output_file}")

    # Comparison table
    print()
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Valid%':>8} {'Power%':>8} {'NestedTrig%':>12} {'AvgDepth':>10}")
    print("-"*80)
    for model_name, data in results.items():
        stats = data["stats"]
        print(f"{model_name:<20} {stats['valid_pct']:>7.1f}% {stats['has_power_pct']:>7.1f}% {stats['has_nested_trig_pct']:>11.1f}% {stats['avg_depth']:>10.2f}")


if __name__ == "__main__":
    main()
