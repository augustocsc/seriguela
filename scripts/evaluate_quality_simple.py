#!/usr/bin/env python3
"""
Simple quality evaluation without requiring specific dataset.
Generates expressions with random prompts and measures validity.
"""

import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.expression import Expression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common variables and operators for symbolic regression
COMMON_VARS = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5']
COMMON_OPS = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tan']


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


def create_random_prompt():
    """Create a random JSON prompt for expression generation"""
    num_vars = random.randint(1, 3)
    num_ops = random.randint(3, 7)

    vars_list = random.sample(COMMON_VARS, num_vars)
    ops_list = random.sample(COMMON_OPS, num_ops)

    prompt = {
        "vars": vars_list,
        "ops": ops_list,
        "cons": "C",
        "expr": ""
    }

    prompt_str = json.dumps(prompt, ensure_ascii=False)
    prompt_str = prompt_str.rsplit('"expr":', 1)[0] + '"expr": "'
    return prompt_str, vars_list, ops_list


def extract_expression_json(output: str):
    """Extract expression from JSON output"""
    import re

    match = re.search(r'"expr":\s*"([^"]*)"', output)
    if match:
        return match.group(1)

    match = re.search(r'"expr":\s*"([^"]+)', output)
    if match:
        expr = match.group(1)
        expr = expr.split('"')[0].split('}')[0].strip()
        return expr

    return None


def evaluate_model(model, tokenizer, num_samples=500):
    """Evaluate model on random prompts"""
    device = model.device

    results = []
    valid_count = 0
    parseable_count = 0
    unique_expressions = set()

    random.seed(42)

    logger.info(f"Evaluating on {num_samples} random prompts...")

    for i in tqdm(range(num_samples), desc="Generating"):
        prompt, vars_list, ops_list = create_random_prompt()

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
        is_parseable = False
        error_msg = None

        if expr_str:
            try:
                expr = Expression(expr_str, is_prefix=False)
                is_parseable = True
                # Expression doesn't have validate() method, check if it was created successfully
                is_valid = is_parseable and expr.sympy_expression is not None
                if is_valid:
                    unique_expressions.add(expr_str)
            except Exception as e:
                error_msg = str(e)[:100]
        else:
            error_msg = "Failed to extract expression"

        if is_valid:
            valid_count += 1
        if is_parseable:
            parseable_count += 1

        results.append({
            "sample_idx": i,
            "prompt": prompt[:200],
            "generated": generated[:500],
            "expression": expr_str,
            "valid": is_valid,
            "parseable": is_parseable,
            "error": error_msg
        })

    total = len(results)
    metrics = {
        "num_samples": total,
        "valid_rate": valid_count / total if total > 0 else 0,
        "parseable_rate": parseable_count / total if total > 0 else 0,
        "unique_expressions": len(unique_expressions),
        "diversity_rate": len(unique_expressions) / total if total > 0 else 0,
    }

    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    model, tokenizer, base_model_name = load_model_auto(args.model_path)
    metrics, results = evaluate_model(model, tokenizer, args.num_samples)

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {os.path.basename(args.model_path)}")
    print("="*60)
    print(f"Base model: {base_model_name}")
    print(f"Valid rate: {metrics['valid_rate']*100:.1f}%")
    print(f"Parseable rate: {metrics['parseable_rate']*100:.1f}%")
    print(f"Unique expressions: {metrics['unique_expressions']}")
    print(f"Diversity rate: {metrics['diversity_rate']*100:.1f}%")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path)

    metrics_path = os.path.join(args.output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    results_path = os.path.join(args.output_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
