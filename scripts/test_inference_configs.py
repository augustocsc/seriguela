"""
Test different inference configurations to find optimal generation parameters.

This script tests various combinations of:
- Temperature (sampling randomness)
- Top-k and top-p (nucleus sampling)
- Repetition penalty
- Max length
- Stopping criteria

Usage:
    python scripts/test_inference_configs.py \
        --model_path ./output/Se124M_700K_infix_v3 \
        --num_samples 20 \
        --output_dir ./inference_tests/v3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExpressionStoppingCriteria(StoppingCriteria):
    """Stop generation at <|endofex|> token."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.end_token_id = tokenizer.encode("<|endofex|>", add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if we've generated the end token
        if input_ids.shape[1] <= self.prompt_length:
            return False

        # Check last few tokens for end marker
        recent_tokens = input_ids[0, -len(self.end_token_id):].tolist()
        return recent_tokens == self.end_token_id


# Inference configurations to test
INFERENCE_CONFIGS = {
    "default": {
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "do_sample": True,
        "description": "Default transformers settings"
    },
    "greedy": {
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "do_sample": False,
        "description": "Greedy decoding (no sampling)"
    },
    "low_temp": {
        "temperature": 0.3,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "do_sample": True,
        "description": "Low temperature for more focused output"
    },
    "high_temp": {
        "temperature": 1.5,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "do_sample": True,
        "description": "Higher temperature for more diversity"
    },
    "nucleus_strict": {
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "do_sample": True,
        "description": "Strict nucleus sampling (top-p=0.8)"
    },
    "nucleus_relaxed": {
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "do_sample": True,
        "description": "Relaxed nucleus sampling (top-p=0.95)"
    },
    "with_repetition_penalty": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "max_new_tokens": 128,
        "do_sample": True,
        "description": "With repetition penalty to avoid loops"
    },
    "strong_repetition_penalty": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.5,
        "max_new_tokens": 128,
        "do_sample": True,
        "description": "Strong repetition penalty"
    },
    "short_generation": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 64,
        "do_sample": True,
        "description": "Shorter max length (64 tokens)"
    },
    "optimized": {
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_new_tokens": 100,
        "do_sample": True,
        "description": "Optimized settings (balanced)"
    },
}


def load_model_and_tokenizer(model_path: str, base_model: str = "gpt2"):
    """Load model and tokenizer, handling both base and LoRA models."""
    logger.info(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Add special tokens if not present
    special_tokens = {
        "additional_special_tokens": ["<|startofex|>", "<|endofex|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    # Try loading as LoRA model first
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        base.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        logger.info("Loaded as LoRA model and merged")
    except Exception as e:
        # Load as regular model
        logger.info(f"Loading as regular model (not LoRA): {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    model.eval()
    logger.info(f"Model loaded on: {model.device}")
    return model, tokenizer


def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    use_stopping_criteria: bool = True
) -> tuple[str, Dict[str, Any]]:
    """Generate text with specific configuration."""

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    # Setup stopping criteria
    stopping_criteria = None
    if use_stopping_criteria:
        stopping_criteria = StoppingCriteriaList([
            ExpressionStoppingCriteria(tokenizer, prompt_length)
        ])

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **{k: v for k, v in config.items() if k != "description"},
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_time = time.time() - start_time

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the generated part
    generated_only = tokenizer.decode(
        outputs[0][prompt_length:],
        skip_special_tokens=False
    )

    # Statistics
    stats = {
        "total_tokens": outputs.shape[1],
        "generated_tokens": outputs.shape[1] - prompt_length,
        "generation_time": generation_time,
        "tokens_per_second": (outputs.shape[1] - prompt_length) / generation_time,
    }

    return generated_only, stats


def extract_expression(generated_text: str) -> tuple[str, str]:
    """Extract expression from generated text."""

    # Strategy 1: Look for <|endofex|> marker
    if "<|endofex|>" in generated_text:
        expr = generated_text.split("<|endofex|>")[0].strip()
        # Remove "expr:" prefix if present
        if "expr:" in expr:
            expr = expr.split("expr:")[-1].strip()
        return expr, "marker"

    # Strategy 2: Look for expr: prefix
    if "expr:" in generated_text:
        parts = generated_text.split("expr:")
        if len(parts) > 1:
            # Take until newline or vars:
            expr = parts[1].split("\n")[0].strip()
            expr = expr.split("vars:")[0].strip()
            return expr, "prefix"

    # Strategy 3: Take first line
    first_line = generated_text.split("\n")[0].strip()
    if first_line:
        return first_line, "first_line"

    return generated_text.strip(), "raw"


def validate_expression(expr: str) -> Dict[str, Any]:
    """Simple validation of expression quality."""
    issues = []

    # Check for repetition
    if len(expr) > 10:
        for i in range(len(expr) - 5):
            substr = expr[i:i+3]
            if expr.count(substr) > 3:
                issues.append(f"repetition: '{substr}'")
                break

    # Check for concatenation
    if "<|endofex|>" in expr:
        issues.append("marker_in_expression")

    # Check for garbage tokens
    garbage_tokens = [
        "Buyable", "Instore", "AndOnline", "Store", "Online",
        "Product", "Available", "Shopping"
    ]
    for token in garbage_tokens:
        if token in expr:
            issues.append(f"garbage: {token}")

    # Check for valid math operators
    valid_operators = ["sin", "cos", "tan", "log", "exp", "sqrt", "abs", "+", "-", "*", "/", "**"]
    has_operator = any(op in expr for op in valid_operators)

    # Check for variables
    has_variable = any(f"x_{i}" in expr or f"C" in expr for i in range(1, 20))

    return {
        "is_valid": len(issues) == 0 and has_operator and has_variable,
        "has_operator": has_operator,
        "has_variable": has_variable,
        "issues": issues,
        "length": len(expr),
    }


def test_configurations(
    model,
    tokenizer,
    test_prompts: List[str],
    output_dir: Path,
    configs_to_test: List[str] = None
):
    """Test all configurations on test prompts."""

    if configs_to_test is None:
        configs_to_test = list(INFERENCE_CONFIGS.keys())

    results = []

    logger.info(f"\nTesting {len(configs_to_test)} configurations on {len(test_prompts)} prompts...")

    for config_name in configs_to_test:
        config = INFERENCE_CONFIGS[config_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing config: {config_name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*60}")

        config_results = []

        for i, prompt in enumerate(test_prompts):
            logger.info(f"\nPrompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")

            # Test with stopping criteria
            try:
                generated, stats = generate_with_config(
                    model, tokenizer, prompt, config, use_stopping_criteria=True
                )

                # Extract expression
                expr, extraction_method = extract_expression(generated)

                # Validate
                validation = validate_expression(expr)

                result = {
                    "config_name": config_name,
                    "config_description": config["description"],
                    "prompt": prompt,
                    "generated_raw": generated[:200],
                    "expression": expr[:200],
                    "extraction_method": extraction_method,
                    "is_valid": validation["is_valid"],
                    "has_operator": validation["has_operator"],
                    "has_variable": validation["has_variable"],
                    "issues": ", ".join(validation["issues"]) if validation["issues"] else "",
                    "expr_length": validation["length"],
                    "total_tokens": stats["total_tokens"],
                    "generated_tokens": stats["generated_tokens"],
                    "generation_time": stats["generation_time"],
                    "tokens_per_second": stats["tokens_per_second"],
                }

                config_results.append(result)
                results.append(result)

                # Log result
                status = "✅ VALID" if validation["is_valid"] else "❌ INVALID"
                logger.info(f"  {status}: {expr[:80]}")
                if validation["issues"]:
                    logger.info(f"  Issues: {', '.join(validation['issues'])}")

            except Exception as e:
                logger.error(f"Error generating with {config_name}: {e}")
                results.append({
                    "config_name": config_name,
                    "config_description": config["description"],
                    "prompt": prompt,
                    "error": str(e),
                    "is_valid": False,
                })

        # Summary for this config
        valid_count = sum(1 for r in config_results if r.get("is_valid", False))
        valid_rate = valid_count / len(config_results) * 100 if config_results else 0

        avg_tokens = sum(r.get("generated_tokens", 0) for r in config_results) / len(config_results) if config_results else 0
        avg_time = sum(r.get("generation_time", 0) for r in config_results) / len(config_results) if config_results else 0

        logger.info(f"\n{'='*60}")
        logger.info(f"Config {config_name} Summary:")
        logger.info(f"  Valid: {valid_count}/{len(config_results)} ({valid_rate:.1f}%)")
        logger.info(f"  Avg tokens: {avg_tokens:.1f}")
        logger.info(f"  Avg time: {avg_time:.3f}s")
        logger.info(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test different inference configurations"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (local or HuggingFace Hub)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="Base model for LoRA"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of test prompts to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Specific configs to test (default: all)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    # Create test prompts
    test_prompts = [
        "vars: x_1, x_2, x_3\noper: *, +, -, sin, cos, log\ncons: C\nexpr:",
        "vars: x_1, x_2\noper: *, **, exp, log\ncons: C\nexpr:",
        "vars: x_1, x_2, x_3, x_4\noper: *, +, /, sqrt, abs\ncons: C\nexpr:",
        "vars: x_1\noper: sin, cos, exp\ncons: C\nexpr:",
        "vars: x_1, x_2, x_3\noper: *, +, -, tan\ncons: C\nexpr:",
    ] * (args.num_samples // 5 + 1)
    test_prompts = test_prompts[:args.num_samples]

    # Test configurations
    results = test_configurations(
        model,
        tokenizer,
        test_prompts,
        output_dir,
        args.configs
    )

    # Save detailed results
    df = pd.DataFrame(results)
    results_file = output_dir / "inference_config_results.csv"
    df.to_csv(results_file, index=False)
    logger.info(f"\nDetailed results saved to: {results_file}")

    # Generate summary report
    summary = {}
    for config_name in df["config_name"].unique():
        config_df = df[df["config_name"] == config_name]
        summary[config_name] = {
            "description": config_df["config_description"].iloc[0] if len(config_df) > 0 else "",
            "valid_rate": (config_df["is_valid"].sum() / len(config_df) * 100) if len(config_df) > 0 else 0,
            "total_samples": len(config_df),
            "valid_count": int(config_df["is_valid"].sum()),
            "avg_tokens": float(config_df["generated_tokens"].mean()) if "generated_tokens" in config_df else 0,
            "avg_time": float(config_df["generation_time"].mean()) if "generation_time" in config_df else 0,
            "common_issues": config_df["issues"].value_counts().head(3).to_dict() if "issues" in config_df else {},
        }

    # Sort by valid rate
    summary = dict(sorted(summary.items(), key=lambda x: x[1]["valid_rate"], reverse=True))

    summary_file = output_dir / "inference_config_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    for config_name, stats in summary.items():
        logger.info(f"\n{config_name}:")
        logger.info(f"  Description: {stats['description']}")
        logger.info(f"  Valid rate: {stats['valid_rate']:.1f}% ({stats['valid_count']}/{stats['total_samples']})")
        logger.info(f"  Avg tokens: {stats['avg_tokens']:.1f}")
        logger.info(f"  Avg time: {stats['avg_time']:.3f}s")

    logger.info("\n" + "="*60)
    logger.info("Testing complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
