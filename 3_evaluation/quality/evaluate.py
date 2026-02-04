# Script para avaliacao customizada de modelos treinados
# Projeto Seriguela - Avaliacao de expressoes simbolicas geradas

import argparse
import json
import os
import sys
import re
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.expression import Expression


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on expression generation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (local or HuggingFace Hub)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model for PEFT (if model_path is adapter)")
    parser.add_argument("--dataset_repo_id", type=str, default="augustocsc/sintetico_natural",
                        help="HuggingFace dataset repository")
    parser.add_argument("--data_dir", type=str, default="700K",
                        help="Data directory within dataset")
    parser.add_argument("--data_column", type=str, default="i_prompt_n",
                        help="Column name for prompts (i_prompt_n for infix, p_prompt_n for prefix)")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--num_generations", type=int, default=1,
                        help="Number of generations per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    return parser.parse_args()


def extract_expression_from_output(output: str, is_prefix: bool = False) -> str:
    """Extract the expression from model output."""
    # Try marker-based first
    start_marker = "<|startofex|>"
    end_marker = "<|endofex|>"

    if start_marker in output and end_marker in output:
        start_idx = output.find(start_marker) + len(start_marker)
        end_idx = output.find(end_marker)
        if start_idx < end_idx:
            return output[start_idx:end_idx].strip()

    # Fallback: Extract first complete expression after start marker
    if start_marker in output:
        start_idx = output.find(start_marker) + len(start_marker)
        remaining = output[start_idx:].strip()

        # Split at common boundaries
        for boundary in ["\nvars:", "\nVariables:", "\nOperators:", "\n\n", "<|endoftext|>"]:
            if boundary in remaining:
                remaining = remaining.split(boundary)[0].strip()
                break

        # Remove any trailing incomplete text - take just the first line
        remaining = remaining.split("\n")[0].strip()

        # Limit length if unreasonably long
        if len(remaining) > 150:
            remaining = remaining[:150]

        return remaining

    # Last resort: look for "expr:" or "Expression:" pattern
    match = re.search(r'(?:expr|Expression):\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Give up: return first line, limited length
    first_line = output.strip().split("\n")[0]
    return first_line[:100] if len(first_line) > 100 else first_line


def validate_expression(expr_str: str, is_prefix: bool = False) -> dict:
    """Validate if expression is syntactically correct."""
    result = {
        "valid": False,
        "parseable": False,
        "error": None,
        "expression_obj": None
    }

    if not expr_str or expr_str.strip() == "":
        result["error"] = "Empty expression"
        return result

    try:
        expr_obj = Expression(expr_str, is_prefix=is_prefix)
        result["parseable"] = True
        result["valid"] = True
        result["expression_obj"] = expr_obj
    except Exception as e:
        result["error"] = str(e)

    return result


def check_prompt_adherence(expr_str: str, prompt: str, is_prefix: bool = False) -> dict:
    """Check if expression adheres to prompt constraints."""
    result = {
        "uses_allowed_vars": False,
        "uses_allowed_ops": False,
        "all_constraints_met": False
    }

    # Extract allowed vars and ops from prompt
    # Typical prompt format: "Variables: x_1, x_2, x_3\nOperators: +, -, *, sin\n..."

    # Extract variables from prompt
    var_match = re.search(r"Variables?:\s*([^\n]+)", prompt, re.IGNORECASE)
    allowed_vars = set()
    if var_match:
        var_str = var_match.group(1)
        # Match patterns like x_1, x_2, etc.
        allowed_vars = set(re.findall(r"x_\d+", var_str))

    # Extract operators from prompt
    op_match = re.search(r"Operators?:\s*([^\n]+)", prompt, re.IGNORECASE)
    allowed_ops = set()
    if op_match:
        op_str = op_match.group(1)
        # Common operators
        ops = ['+', '-', '*', '/', '**', 'sin', 'cos', 'tan', 'log', 'sqrt', 'exp']
        for op in ops:
            if op in op_str:
                allowed_ops.add(op)

    # Check variables in expression
    expr_vars = set(re.findall(r"x_\d+", expr_str))
    if allowed_vars:
        result["uses_allowed_vars"] = expr_vars.issubset(allowed_vars)
    else:
        result["uses_allowed_vars"] = True  # No constraint specified

    # Check operators (simplified check)
    result["uses_allowed_ops"] = True  # Default to true if no ops specified
    if allowed_ops:
        # This is a simplified check - would need more sophisticated parsing for accuracy
        for op in ['sin', 'cos', 'tan', 'log', 'sqrt', 'exp']:
            if op in expr_str and op not in allowed_ops:
                result["uses_allowed_ops"] = False
                break

    result["all_constraints_met"] = result["uses_allowed_vars"] and result["uses_allowed_ops"]

    return result


def load_model_and_tokenizer(model_path: str, base_model: str = None, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if this is a PEFT model
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json")) if os.path.isdir(model_path) else False

    if is_peft or base_model:
        # Load base model first
        base = base_model or "gpt2"
        print(f"Loading base model: {base}")
        model = AutoModelForCausalLM.from_pretrained(base)
        model.resize_token_embeddings(len(tokenizer))

        # Load PEFT adapter
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_expression(model, tokenizer, prompt: str, device: str,
                        max_new_tokens: int = 128, temperature: float = 0.7,
                        top_p: float = 0.9, num_return_sequences: int = 1):
    """Generate expression(s) from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return generated


def evaluate_model(args):
    """Main evaluation function."""
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, args.base_model, args.device
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset_repo_id}/{args.data_dir}")
    try:
        dataset = load_dataset(
            args.dataset_repo_id,
            data_files={
                "test": f"{args.data_dir}/test_{args.data_dir}.csv"
            }
        )["test"]
    except Exception as e:
        print(f"Error loading test set, trying validation: {e}")
        dataset = load_dataset(
            args.dataset_repo_id,
            data_files={
                "validation": f"{args.data_dir}/val_{args.data_dir}.csv"
            }
        )["validation"]

    # Sample if needed
    if len(dataset) > args.num_samples:
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = dataset.select(indices)

    print(f"Evaluating on {len(dataset)} samples...")

    # Determine if prefix or infix
    is_prefix = args.data_column.startswith("p_")

    # Evaluation metrics
    metrics = {
        "total_samples": 0,
        "total_generations": 0,
        "valid_expressions": 0,
        "parseable_expressions": 0,
        "uses_allowed_vars": 0,
        "uses_allowed_ops": 0,
        "all_constraints_met": 0,
        "unique_expressions": set(),
        "expression_lengths": [],
        "errors": Counter(),
    }

    results = []

    # Generate and evaluate
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = sample[args.data_column]

        # Extract just the prompt part (before the expression)
        # Typically the prompt ends before <|startofex|>
        if "<|startofex|>" in prompt:
            prompt_only = prompt.split("<|startofex|>")[0] + "<|startofex|>"
        else:
            prompt_only = prompt

        generations = generate_expression(
            model, tokenizer, prompt_only, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_generations
        )

        metrics["total_samples"] += 1

        for gen_output in generations:
            metrics["total_generations"] += 1

            # Extract expression
            expr_str = extract_expression_from_output(gen_output, is_prefix)

            # Validate
            validation = validate_expression(expr_str, is_prefix)

            # Check adherence
            adherence = check_prompt_adherence(expr_str, prompt_only, is_prefix)

            # Update metrics
            if validation["valid"]:
                metrics["valid_expressions"] += 1
            if validation["parseable"]:
                metrics["parseable_expressions"] += 1
                metrics["unique_expressions"].add(expr_str)
                metrics["expression_lengths"].append(len(expr_str))
            if validation["error"]:
                metrics["errors"][validation["error"][:50]] += 1

            if adherence["uses_allowed_vars"]:
                metrics["uses_allowed_vars"] += 1
            if adherence["uses_allowed_ops"]:
                metrics["uses_allowed_ops"] += 1
            if adherence["all_constraints_met"]:
                metrics["all_constraints_met"] += 1

            results.append({
                "sample_idx": idx,
                "prompt": prompt_only[:200],  # Truncate for storage
                "generated_output": gen_output[:500],
                "extracted_expression": expr_str,
                "valid": validation["valid"],
                "parseable": validation["parseable"],
                "error": validation["error"],
                "uses_allowed_vars": adherence["uses_allowed_vars"],
                "uses_allowed_ops": adherence["uses_allowed_ops"],
            })

    # Calculate final metrics
    total_gen = metrics["total_generations"]
    final_metrics = {
        "model_path": args.model_path,
        "dataset": f"{args.dataset_repo_id}/{args.data_dir}",
        "data_column": args.data_column,
        "is_prefix": is_prefix,
        "num_samples": metrics["total_samples"],
        "num_generations": total_gen,
        "temperature": args.temperature,
        "top_p": args.top_p,

        # Validity metrics
        "valid_rate": metrics["valid_expressions"] / total_gen if total_gen > 0 else 0,
        "parseable_rate": metrics["parseable_expressions"] / total_gen if total_gen > 0 else 0,

        # Adherence metrics
        "uses_allowed_vars_rate": metrics["uses_allowed_vars"] / total_gen if total_gen > 0 else 0,
        "uses_allowed_ops_rate": metrics["uses_allowed_ops"] / total_gen if total_gen > 0 else 0,
        "constraints_met_rate": metrics["all_constraints_met"] / total_gen if total_gen > 0 else 0,

        # Diversity metrics
        "unique_expressions": len(metrics["unique_expressions"]),
        "diversity_rate": len(metrics["unique_expressions"]) / total_gen if total_gen > 0 else 0,
        "avg_expression_length": np.mean(metrics["expression_lengths"]) if metrics["expression_lengths"] else 0,

        # Error distribution (top 10)
        "top_errors": dict(metrics["errors"].most_common(10)),

        "timestamp": datetime.now().isoformat(),
    }

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_repo_id}/{args.data_dir}")
    print(f"Format: {'Prefix' if is_prefix else 'Infix'}")
    print("-"*60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Total generations: {total_gen}")
    print("-"*60)
    print("VALIDITY METRICS:")
    print(f"  Valid rate: {final_metrics['valid_rate']:.2%}")
    print(f"  Parseable rate: {final_metrics['parseable_rate']:.2%}")
    print("-"*60)
    print("ADHERENCE METRICS:")
    print(f"  Uses allowed vars: {final_metrics['uses_allowed_vars_rate']:.2%}")
    print(f"  Uses allowed ops: {final_metrics['uses_allowed_ops_rate']:.2%}")
    print(f"  All constraints met: {final_metrics['constraints_met_rate']:.2%}")
    print("-"*60)
    print("DIVERSITY METRICS:")
    print(f"  Unique expressions: {final_metrics['unique_expressions']}")
    print(f"  Diversity rate: {final_metrics['diversity_rate']:.2%}")
    print(f"  Avg expression length: {final_metrics['avg_expression_length']:.1f}")
    print("="*60)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Create filename from model path
    model_name = args.model_path.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics
    metrics_file = os.path.join(args.output_dir, f"metrics_{model_name}_{timestamp}.json")
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save detailed results
    results_file = os.path.join(args.output_dir, f"results_{model_name}_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")

    return final_metrics


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
