#!/usr/bin/env python3
"""
Complete evaluation and finetuning pipeline for all 3 prefix models.
1. Compare Base vs Medium vs Large (50 samples each)
2. Run RL finetuning on best model (Nguyen-5)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from collections import Counter
import re
import sys
import os

def load_model(model_path):
    """Load LoRA model."""
    print(f"Loading model from {model_path}...")

    # Determine base model from path
    if "large" in model_path.lower():
        base_model_name = "gpt2-large"
    elif "medium" in model_path.lower():
        base_model_name = "gpt2-medium"
    else:
        base_model_name = "gpt2"

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    print(f"✓ Model loaded: {base_model_name} + LoRA adapter")
    return model, tokenizer, base_model_name

def generate_expressions(model, tokenizer, num_samples=50, temperature=0.8):
    """Generate expressions from model."""
    print(f"Generating {num_samples} expressions...")

    prompt = '{"vars": ["x_1"], "ops": ["*", "+", "-", "sin", "cos", "exp", "pow"], "cons": "C", "expr": "'

    expressions = []

    for i in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract expression
        match = re.search(r'"expr":\s*"([^"]+)"', generated)
        if match:
            expr = match.group(1)
            expressions.append(expr)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples}...")

    print(f"✓ Generated {len(expressions)} expressions")
    return expressions

def analyze_expressions(expressions):
    """Analyze expression complexity."""
    stats = {
        "total": len(expressions),
        "unique": len(set(expressions)),
        "diversity_rate": len(set(expressions)) / len(expressions) * 100 if expressions else 0,
        "with_power": 0,
        "with_trig": 0,
        "with_nested_trig": 0,
        "operators": Counter(),
        "examples": expressions[:10]
    }

    for expr in expressions:
        if '**' in expr or 'pow(' in expr:
            stats["with_power"] += 1

        if 'sin' in expr or 'cos' in expr:
            stats["with_trig"] += 1

        if 'sin(sin' in expr or 'sin(cos' in expr or 'cos(sin' in expr or 'cos(cos' in expr:
            stats["with_nested_trig"] += 1

        for op in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'pow', '**']:
            if op in expr:
                stats["operators"][op] += 1

    return stats

def compare_models():
    """Compare all three models."""
    print("="*80)
    print("PHASE 1: Model Comparison (Base vs Medium vs Large)")
    print("="*80)
    print()

    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available, using CPU (very slow)")
        print()

    models_to_compare = [
        ("base", "./output/gpt2_base_prefix_682k"),
        ("medium", "./output/gpt2_medium_prefix_682k"),
        ("large", "./output/gpt2_large_prefix_682k")
    ]

    all_results = {}

    for name, path in models_to_compare:
        if not os.path.exists(path):
            print(f"⚠️  {name.upper()} model not found at {path}, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating {name.upper()} ({path})")
        print('='*80)

        model, tokenizer, base_name = load_model(path)
        expressions = generate_expressions(model, tokenizer, num_samples=50)
        stats = analyze_expressions(expressions)

        all_results[name] = {
            "base_model": base_name,
            "expressions": expressions,
            "stats": {k: v for k, v in stats.items() if k != "operators"},
            "operators": dict(stats["operators"])
        }

        # Free memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"✓ {name.upper()} evaluation complete")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print()

    if len(all_results) >= 2:
        print("| Metric | Base | Medium | Large |")
        print("|--------|------|--------|-------|")

        metrics = [
            ("Total", "total"),
            ("Unique", "unique"),
            ("Diversity (%)", "diversity_rate"),
            ("With power (%)", lambda s: f"{s['with_power']/s['total']*100:.1f}"),
            ("With trig (%)", lambda s: f"{s['with_trig']/s['total']*100:.1f}"),
            ("Nested trig (%)", lambda s: f"{s['with_nested_trig']/s['total']*100:.1f}")
        ]

        for metric_name, metric_key in metrics:
            row = f"| {metric_name} |"
            for model_name in ["base", "medium", "large"]:
                if model_name in all_results:
                    stats = all_results[model_name]["stats"]
                    if callable(metric_key):
                        value = metric_key(stats)
                    else:
                        value = stats.get(metric_key, "-")
                    row += f" {value} |"
                else:
                    row += " - |"
            print(row)

    # Save results
    output_file = "comparison_all_models_prefix.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Comparison results saved to: {output_file}")
    print()

    return all_results

def run_rl_finetuning(best_model_name):
    """Run RL finetuning on best model (Nguyen-5)."""
    print("="*80)
    print("PHASE 2: RL Finetuning (REINFORCE on Nguyen-5)")
    print("="*80)
    print()

    model_path = f"./output/gpt2_{best_model_name}_prefix_682k"

    if not os.path.exists("scripts/reinforce_symbolic.py"):
        print("⚠️  REINFORCE script not found, skipping finetuning")
        print("   This is optional - evaluation is complete")
        return

    print(f"Running REINFORCE finetuning on {best_model_name.upper()}...")
    print("Dataset: Nguyen-5 (sin(x_1**2)*cos(x_1) - 1)")
    print("Epochs: 20")
    print()

    # Run REINFORCE
    cmd = (
        f"/opt/pytorch/bin/python3 scripts/reinforce_symbolic.py "
        f"--model_path {model_path} "
        f"--dataset data/benchmarks/nguyen/nguyen_5.csv "
        f"--epochs 20 "
        f"--output_dir ./output/rl_finetuned_{best_model_name} "
        f"--output_file rl_results_{best_model_name}.json"
    )

    print(f"Command: {cmd}")
    print()

    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        print(result.stdout)
        if result.returncode != 0:
            print("⚠️  REINFORCE failed:")
            print(result.stderr)
            return False
        print("✓ RL finetuning complete")
        return True
    except subprocess.TimeoutExpired:
        print("⚠️  REINFORCE timeout (2 hours)")
        return False
    except Exception as e:
        print(f"⚠️  REINFORCE error: {e}")
        return False

def main():
    print("="*80)
    print("Complete Evaluation and Finetuning Pipeline")
    print("="*80)
    print()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available - will use CPU (slow)")
    print()

    # Phase 1: Compare all models
    results = compare_models()

    # Determine best model
    best_model = None
    best_score = 0

    for name, data in results.items():
        stats = data["stats"]
        # Score = power_ops + trig + diversity
        score = (stats["with_power"]/stats["total"]*100 +
                 stats["with_trig"]/stats["total"]*100 +
                 stats["diversity_rate"])
        print(f"{name.upper()} score: {score:.1f}")

        if score > best_score:
            best_score = score
            best_model = name

    if best_model:
        print(f"\n✓ Best model: {best_model.upper()} (score: {best_score:.1f})")
        print()

        # Phase 2: RL finetuning (optional)
        print("Starting RL finetuning phase...")
        run_rl_finetuning(best_model)
    else:
        print("⚠️  No models evaluated, skipping finetuning")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print()
    print("Results saved:")
    print("  - comparison_all_models_prefix.json")
    print("  - rl_results_*.json (if RL ran)")
    print()

if __name__ == "__main__":
    main()
