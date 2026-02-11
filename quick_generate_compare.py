#!/usr/bin/env python3
"""
Quick script to generate and compare expressions from Base and Medium prefix models.
Bypasses import issues by using minimal dependencies.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from collections import Counter
import re

def load_model(model_path):
    """Load LoRA model."""
    print(f"Loading model from {model_path}...")

    # Determine base model from path
    if "base" in model_path.lower():
        base_model_name = "gpt2"
    elif "medium" in model_path.lower():
        base_model_name = "gpt2-medium"
    elif "large" in model_path.lower():
        base_model_name = "gpt2-large"
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
    return model, tokenizer

def generate_expressions(model, tokenizer, num_samples=50, temperature=0.8):
    """Generate expressions from model."""
    print(f"Generating {num_samples} expressions...")

    # Prefix notation prompt
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

        # Extract expression (between "expr": " and ")
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
        # Check for power
        if '**' in expr or 'pow(' in expr:
            stats["with_power"] += 1

        # Check for trig
        if 'sin' in expr or 'cos' in expr:
            stats["with_trig"] += 1

        # Check for nested trig
        if 'sin(sin' in expr or 'sin(cos' in expr or 'cos(sin' in expr or 'cos(cos' in expr:
            stats["with_nested_trig"] += 1

        # Count operators
        for op in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'pow', '**']:
            if op in expr:
                stats["operators"][op] += 1

    return stats

def main():
    print("="*80)
    print("Quick Model Comparison: Base vs Medium (Prefix Notation)")
    print("="*80)
    print()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU (slow)")
    print()

    # Load models
    base_model, base_tokenizer = load_model("./output/gpt2_base_prefix_682k")
    print()

    medium_model, medium_tokenizer = load_model("./output/gpt2_medium_prefix_682k")
    print()

    # Generate expressions
    print("="*80)
    print("Generating from Base (124M)")
    print("="*80)
    base_expressions = generate_expressions(base_model, base_tokenizer, num_samples=50)
    print()

    print("="*80)
    print("Generating from Medium (355M)")
    print("="*80)
    medium_expressions = generate_expressions(medium_model, medium_tokenizer, num_samples=50)
    print()

    # Analyze
    print("="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print()

    base_stats = analyze_expressions(base_expressions)
    medium_stats = analyze_expressions(medium_expressions)

    print("| Metric | Base (124M) | Medium (355M) | Difference |")
    print("|--------|-------------|---------------|------------|")
    print(f"| Total generated | {base_stats['total']} | {medium_stats['total']} | - |")
    print(f"| Unique expressions | {base_stats['unique']} | {medium_stats['unique']} | {medium_stats['unique'] - base_stats['unique']:+d} |")
    print(f"| Diversity rate (%) | {base_stats['diversity_rate']:.1f} | {medium_stats['diversity_rate']:.1f} | {medium_stats['diversity_rate'] - base_stats['diversity_rate']:+.1f} |")
    print(f"| With power ops (%) | {base_stats['with_power']/base_stats['total']*100:.1f} | {medium_stats['with_power']/medium_stats['total']*100:.1f} | {(medium_stats['with_power']/medium_stats['total'] - base_stats['with_power']/base_stats['total'])*100:+.1f} |")
    print(f"| With trig (%) | {base_stats['with_trig']/base_stats['total']*100:.1f} | {medium_stats['with_trig']/medium_stats['total']*100:.1f} | {(medium_stats['with_trig']/medium_stats['total'] - base_stats['with_trig']/base_stats['total'])*100:+.1f} |")
    print(f"| With nested trig (%) | {base_stats['with_nested_trig']/base_stats['total']*100:.1f} | {medium_stats['with_nested_trig']/medium_stats['total']*100:.1f} | {(medium_stats['with_nested_trig']/medium_stats['total'] - base_stats['with_nested_trig']/base_stats['total'])*100:+.1f} |")
    print()

    print("="*80)
    print("SAMPLE EXPRESSIONS")
    print("="*80)
    print()
    print("Base (first 5):")
    for i, expr in enumerate(base_stats['examples'][:5], 1):
        print(f"  {i}. {expr}")
    print()

    print("Medium (first 5):")
    for i, expr in enumerate(medium_stats['examples'][:5], 1):
        print(f"  {i}. {expr}")
    print()

    # Save results
    results = {
        "base": {
            "expressions": base_expressions,
            "stats": {k: v for k, v in base_stats.items() if k != "operators"},
            "operators": dict(base_stats["operators"])
        },
        "medium": {
            "expressions": medium_expressions,
            "stats": {k: v for k, v in medium_stats.items() if k != "operators"},
            "operators": dict(medium_stats["operators"])
        }
    }

    output_file = "comparison_base_medium_prefix.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
