#!/usr/bin/env python3
"""
Quick evaluation script for JSON-formatted models.
Reads base model from adapter_config.json automatically.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.expression import Expression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_auto(model_path: str):
    """Load model with automatic base model detection from adapter_config.json"""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"No adapter_config.json found in {model_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path", "gpt2")
    logger.info(f"Loading base model: {base_model_name}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer, base_model_name


def create_json_prompt(vars_list, ops_list, cons="C"):
    """Create JSON format prompt"""
    prompt = {
        "vars": vars_list,
        "ops": ops_list,
        "cons": cons,
        "expr": ""
    }
    prompt_str = json.dumps(prompt, ensure_ascii=False)
    prompt_str = prompt_str.rsplit('"expr":', 1)[0] + '"expr": "'
    return prompt_str


def extract_expression_json(output: str):
    """Extract expression from JSON output"""
    import re

    # Try to extract from "expr": "..." pattern
    match = re.search(r'"expr":\s*"([^"]*)"', output)
    if match:
        return match.group(1)

    # Try without closing quote
    match = re.search(r'"expr":\s*"([^"]+)', output)
    if match:
        expr = match.group(1)
        # Clean up common artifacts
        expr = expr.split('"')[0].split('}')[0].strip()
        return expr

    return None


def evaluate_model(model, tokenizer, num_samples=500, dataset_name="augustocsc/sintetico_natural", data_dir="700K"):
    """Evaluate model on dataset"""

    device = model.device
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset {dataset_name}/{data_dir}")
    dataset = load_dataset(dataset_name, data_dir, split="train")

    # Sample
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    results = []
    valid_count = 0
    parseable_count = 0
    unique_expressions = set()

    logger.info(f"Evaluating on {len(indices)} samples...")

    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[idx]
        prompt_text = sample.get("i_prompt_n", "")

        # Parse prompt to extract vars and ops
        vars_line = [l for l in prompt_text.split('\n') if l.startswith('vars:')]
        ops_line = [l for l in prompt_text.split('\n') if l.startswith('oper:')]

        if not vars_line or not ops_line:
            continue

        vars_list = [v.strip() for v in vars_line[0].replace('vars:', '').split(',')]
        ops_list = [o.strip() for o in ops_line[0].replace('oper:', '').split(',')]

        # Create JSON prompt
        prompt = create_json_prompt(vars_list, ops_list)

        # Generate
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

        # Extract expression
        expr_str = extract_expression_json(generated)

        # Validate
        is_valid = False
        is_parseable = False
        error_msg = None

        if expr_str:
            try:
                expr = Expression.parse_infix(expr_str)
                is_parseable = True
                is_valid = expr.validate()
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
            "sample_idx": idx,
            "prompt": prompt,
            "generated": generated[:500],
            "expression": expr_str,
            "valid": is_valid,
            "parseable": is_parseable,
            "error": error_msg
        })

    total = len(results)
    metrics = {
        "model_path": str(model),
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
    parser.add_argument("--output_dir", type=str, default="./results_corrected")
    args = parser.parse_args()

    # Load model
    model, tokenizer, base_model_name = load_model_auto(args.model_path)

    # Evaluate
    metrics, results = evaluate_model(model, tokenizer, args.num_samples)

    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {os.path.basename(args.model_path)}")
    print("="*60)
    print(f"Base model: {base_model_name}")
    print(f"Valid rate: {metrics['valid_rate']*100:.1f}%")
    print(f"Parseable rate: {metrics['parseable_rate']*100:.1f}%")
    print(f"Unique expressions: {metrics['unique_expressions']}")
    print(f"Diversity rate: {metrics['diversity_rate']*100:.1f}%")
    print("="*60)

    # Save results
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
