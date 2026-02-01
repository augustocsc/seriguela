#!/usr/bin/env python3
"""
Evaluation script for expression generation experiments.

Evaluates trained models on:
1. Valid Rate: % expressions that can be parsed and evaluated
2. Stopping Rate: % that stop correctly (contain end marker)
3. Symbol Accuracy: % that use only symbols from prompt
4. Garbage Rate: % with non-mathematical tokens

Usage:
    python scripts/evaluate_experiments.py \
        --model_path ./output/exp_a_json \
        --experiment_type json \
        --num_samples 200 \
        --output_file ./results/exp_a_results.json
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.expression import Expression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Garbage words that indicate model failure
GARBAGE_WORDS = [
    "Buyable", "Instore", "Online", "Stockholm", "Muslims", "crash",
    "Berman", "expressed", "fluent", "Avenger", "repositories",
    "GREEN", "intuition", "records", "xstatics", "xid", "sinmod",
    "Pressure", "XP", "Variables", "Operators", "Constants"
]


class ExpressionStoppingCriteria(StoppingCriteria):
    """Stop generation when end marker is detected."""

    def __init__(self, tokenizer, stop_sequences: List[str]):
        self.tokenizer = tokenizer
        self.stop_ids = []
        for seq in stop_sequences:
            ids = tokenizer.encode(seq, add_special_tokens=False)
            if ids:
                self.stop_ids.append(ids)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for stop_ids in self.stop_ids:
            if len(input_ids[0]) >= len(stop_ids):
                if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                    return True
        return False


def load_model(model_path: str, experiment_type: str) -> Tuple:
    """Load trained model and tokenizer."""
    logger.info(f"Loading model from {model_path}")

    # Load experiment info
    exp_info_path = os.path.join(model_path, "experiment_info.json")
    if os.path.exists(exp_info_path):
        with open(exp_info_path) as f:
            exp_info = json.load(f)
        logger.info(f"Experiment info: {exp_info}")
        use_native_eos = exp_info.get("use_native_eos", False)
    else:
        use_native_eos = (experiment_type == "eos")
        logger.warning("No experiment_info.json found, inferring from experiment_type")

    # Load base model
    logger.info("Loading base GPT-2...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add special tokens if not using native EOS
    if not use_native_eos:
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|startofex|>", "<|endofex|>"]
        })
        model.resize_token_embeddings(len(tokenizer))

    # Load adapter
    logger.info("Loading adapter...")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer, use_native_eos


def create_prompt_json(vars_list: List[str], ops_list: List[str], cons: str = "C") -> str:
    """Create JSON format prompt for generation."""
    prompt = {
        "vars": vars_list,
        "ops": ops_list,
        "cons": cons,
        "expr": ""
    }
    # Return partial JSON to let model complete
    prompt_str = json.dumps(prompt, ensure_ascii=False)
    # Remove closing part: , "expr": ""}
    prompt_str = prompt_str.rsplit('"expr":', 1)[0] + '"expr": "'
    return prompt_str


def create_prompt_eos(vars_list: List[str], ops_list: List[str], cons: str = "C") -> str:
    """Create EOS format prompt for generation."""
    lines = [
        f"vars: {', '.join(vars_list)}",
        f"oper: {', '.join(ops_list)}",
        f"cons: {cons}",
        "expr: "
    ]
    return "\n".join(lines)


def extract_expression_json(output: str) -> Optional[str]:
    """Extract expression from JSON format output."""
    try:
        # Try to extract from complete JSON
        if output.strip().endswith("}"):
            obj = json.loads(output)
            return obj.get("expr", None)
    except:
        pass

    # Try to extract expression between "expr": " and "
    match = re.search(r'"expr":\s*"([^"]*)"', output)
    if match:
        return match.group(1)

    # Try to extract after "expr": "
    match = re.search(r'"expr":\s*"([^"]*)', output)
    if match:
        return match.group(1)

    return None


def extract_expression_eos(output: str, end_marker: str) -> Optional[str]:
    """Extract expression from EOS format output."""
    if "expr:" not in output:
        return None

    # Get everything after expr:
    expr_part = output.split("expr:")[-1].strip()

    # Remove end marker
    if end_marker in expr_part:
        expr_part = expr_part.split(end_marker)[0].strip()

    # Remove any trailing garbage
    expr_part = expr_part.split("\n")[0].strip()

    return expr_part if expr_part else None


def validate_expression(expr_str: str, allowed_vars: set, allowed_ops: set) -> Dict:
    """Validate an expression for correctness."""
    result = {
        "raw": expr_str,
        "is_valid": False,
        "is_parseable": False,
        "uses_correct_symbols": False,
        "has_garbage": False,
        "error": None
    }

    if not expr_str or not expr_str.strip():
        result["error"] = "Empty expression"
        return result

    # Check for garbage words
    for word in GARBAGE_WORDS:
        if word.lower() in expr_str.lower():
            result["has_garbage"] = True
            result["error"] = f"Contains garbage: {word}"
            return result

    # Try to parse expression
    try:
        expr = Expression(expr_str, is_prefix=False)
        result["is_parseable"] = True

        # Try to evaluate
        X_test = [[1.0] * 10]  # Provide enough variables
        eval_result = expr.evaluate(X_test)
        if len(eval_result) > 0:
            val = eval_result[0]
            if val == val and val != float('inf') and val != float('-inf'):
                result["is_valid"] = True

    except Exception as e:
        result["error"] = str(e)[:100]

    # Check symbol correctness
    expr_clean = expr_str.replace(" ", "")

    # Extract used variables
    used_vars = set(re.findall(r'x_\d+', expr_clean))
    used_ops = set()

    for op in ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "asin", "acos", "atan"]:
        if op in expr_clean:
            used_ops.add(op)

    for op in ["+", "-", "*", "/", "**"]:
        if op in expr_clean:
            used_ops.add(op)

    # Check if using allowed symbols
    var_ok = used_vars.issubset(allowed_vars)
    op_ok = used_ops.issubset(allowed_ops)
    result["uses_correct_symbols"] = var_ok and op_ok

    if not var_ok:
        invalid_vars = used_vars - allowed_vars
        result["error"] = f"Invalid vars: {invalid_vars}"

    return result


def generate_and_evaluate(
    model,
    tokenizer,
    experiment_type: str,
    use_native_eos: bool,
    num_samples: int = 100,
    test_prompts: Optional[List[Dict]] = None
) -> Dict:
    """Generate expressions and evaluate quality."""

    if test_prompts is None:
        # Default test prompts
        test_prompts = [
            {"vars": ["x_1", "x_2"], "ops": ["*", "+", "-", "sin", "cos"], "cons": "C"},
            {"vars": ["x_1", "x_2", "x_3"], "ops": ["*", "+", "/", "exp", "log"], "cons": "C"},
            {"vars": ["x_1"], "ops": ["*", "**", "sin", "sqrt"], "cons": "C"},
            {"vars": ["x_1", "x_2", "x_3", "x_4"], "ops": ["*", "+", "-", "/"], "cons": "C"},
        ]

    # Determine end marker and stopping sequences
    if use_native_eos:
        end_marker = "<|endoftext|>"
        stop_sequences = ["<|endoftext|>", "\n\nvars:"]
    else:
        end_marker = "<|endofex|>"
        stop_sequences = ["<|endofex|>", '"}', "\n\nvars:"]

    stopping_criteria = StoppingCriteriaList([
        ExpressionStoppingCriteria(tokenizer, stop_sequences)
    ])

    # Generation config
    gen_config = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    results = {
        "total": 0,
        "valid": 0,
        "parseable": 0,
        "correct_symbols": 0,
        "garbage": 0,
        "stopped_correctly": 0,
        "samples": []
    }

    samples_per_prompt = num_samples // len(test_prompts)

    logger.info(f"Generating {num_samples} samples ({samples_per_prompt} per prompt)...")

    for prompt_config in test_prompts:
        vars_list = prompt_config["vars"]
        ops_list = prompt_config["ops"]
        cons = prompt_config.get("cons", "C")

        allowed_vars = set(vars_list) | {cons}
        allowed_ops = set(ops_list) | {"(", ")"}

        # Create prompt based on experiment type
        if experiment_type == "json":
            prompt = create_prompt_json(vars_list, ops_list, cons)
        else:
            prompt = create_prompt_eos(vars_list, ops_list, cons)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        for i in range(samples_per_prompt):
            results["total"] += 1

            # Generate
            output = model.generate(
                **inputs,
                **gen_config,
                stopping_criteria=stopping_criteria
            )
            output_text = tokenizer.decode(output[0], skip_special_tokens=False)

            # Extract expression
            if experiment_type == "json":
                expr_str = extract_expression_json(output_text)
            else:
                expr_str = extract_expression_eos(output_text, end_marker)

            # Check stopping
            stopped_correctly = end_marker in output_text
            if stopped_correctly:
                results["stopped_correctly"] += 1

            # Validate expression
            if expr_str:
                validation = validate_expression(expr_str, allowed_vars, allowed_ops)

                if validation["is_valid"]:
                    results["valid"] += 1
                if validation["is_parseable"]:
                    results["parseable"] += 1
                if validation["uses_correct_symbols"]:
                    results["correct_symbols"] += 1
                if validation["has_garbage"]:
                    results["garbage"] += 1

                # Store sample
                sample = {
                    "prompt_vars": vars_list,
                    "prompt_ops": ops_list,
                    "expression": expr_str,
                    "stopped_correctly": stopped_correctly,
                    **validation
                }
                results["samples"].append(sample)
            else:
                results["garbage"] += 1
                results["samples"].append({
                    "prompt_vars": vars_list,
                    "prompt_ops": ops_list,
                    "expression": None,
                    "stopped_correctly": stopped_correctly,
                    "is_valid": False,
                    "error": "Could not extract expression"
                })

            # Log progress
            if results["total"] % 20 == 0:
                logger.info(f"Progress: {results['total']}/{num_samples}")

    return results


def print_report(results: Dict, experiment_name: str):
    """Print evaluation report."""
    total = results["total"]

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {experiment_name}")
    print("=" * 60)

    print(f"\nTotal samples: {total}")

    metrics = [
        ("Valid Rate", results["valid"] / total * 100),
        ("Parseable Rate", results["parseable"] / total * 100),
        ("Correct Symbols", results["correct_symbols"] / total * 100),
        ("Stopping Rate", results["stopped_correctly"] / total * 100),
        ("Garbage Rate", results["garbage"] / total * 100),
    ]

    print("\nMetrics:")
    print("-" * 40)
    for name, value in metrics:
        status = "PASS" if (name != "Garbage Rate" and value >= 80) or (name == "Garbage Rate" and value < 5) else "FAIL"
        print(f"  {name:<20s}: {value:6.1f}%  [{status}]")

    # Show sample outputs
    print("\n" + "-" * 40)
    print("Sample Outputs:")
    print("-" * 40)

    valid_samples = [s for s in results["samples"] if s.get("is_valid")]
    invalid_samples = [s for s in results["samples"] if not s.get("is_valid")]

    print("\nValid examples:")
    for sample in valid_samples[:5]:
        expr = sample.get("expression", "N/A")
        vars_str = ", ".join(sample.get("prompt_vars", []))
        print(f"  [{vars_str}] -> {expr}")

    print("\nInvalid examples:")
    for sample in invalid_samples[:5]:
        expr = sample.get("expression", "N/A")
        error = sample.get("error", "Unknown")
        print(f"  {expr[:50]}... | Error: {error}")

    print("\n" + "=" * 60)

    # Summary
    valid_rate = results["valid"] / total * 100
    stopping_rate = results["stopped_correctly"] / total * 100
    garbage_rate = results["garbage"] / total * 100

    success = valid_rate >= 80 and stopping_rate >= 90 and garbage_rate < 5

    print(f"\nOVERALL: {'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expression generation experiments"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--experiment_type", type=str, required=True,
                        choices=["json", "eos"],
                        help="Experiment type (json or eos)")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results JSON")

    args = parser.parse_args()

    # Load model
    model, tokenizer, use_native_eos = load_model(
        args.model_path,
        args.experiment_type
    )

    # Generate and evaluate
    results = generate_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        experiment_type=args.experiment_type,
        use_native_eos=use_native_eos,
        num_samples=args.num_samples
    )

    # Print report
    experiment_name = f"EXP-{'A' if args.experiment_type == 'json' else 'B'} ({args.experiment_type.upper()})"
    print_report(results, experiment_name)

    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        # Remove samples for smaller file
        save_results = {k: v for k, v in results.items() if k != "samples"}
        save_results["sample_count"] = len(results["samples"])
        save_results["valid_samples"] = [s for s in results["samples"] if s.get("is_valid")][:20]
        save_results["invalid_samples"] = [s for s in results["samples"] if not s.get("is_valid")][:20]

        with open(args.output_file, "w") as f:
            json.dump(save_results, f, indent=2)

        logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
