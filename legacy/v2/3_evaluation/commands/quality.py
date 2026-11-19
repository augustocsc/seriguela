"""
Quality evaluation command.

Evaluates model generation quality:
- Valid rate: % of syntactically correct expressions
- Diversity rate: % of unique expressions
- Constraint adherence: % respecting allowed vars/ops
- Complexity statistics
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import yaml

# Add 3_evaluation directory to path for imports
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from core.model_loader import ModelLoader
from core.generator import ExpressionGenerator, GenerationConfig, PromptConfig
from core.extractor import ExpressionExtractor
from core.validator import ExpressionValidator
from core.metrics import MetricsCalculator, SampleResult
from core.storage import ResultStorage

logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_config_from_args(args: argparse.Namespace) -> dict:
    """Build configuration dictionary from CLI arguments."""
    config = {
        "model": {
            "path": args.model,
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_tokens,
            "do_sample": True,
        },
        "prompt": {
            "format": "infix",  # Default to infix
        },
        "evaluation": {
            "num_samples": args.num_samples,
            "seed": getattr(args, "seed", 42),
        },
    }

    # Parse vars if provided
    if hasattr(args, "vars") and args.vars:
        config["prompt"]["vars"] = [v.strip() for v in args.vars.split(",")]

    # Parse ops if provided
    if hasattr(args, "ops") and args.ops:
        config["prompt"]["ops"] = [o.strip() for o in args.ops.split(",")]

    # Detect notation from model name
    model_lower = args.model.lower()
    if "prefix" in model_lower:
        config["prompt"]["format"] = "prefix"

    return config


def execute_quality(args: argparse.Namespace):
    """
    Execute quality evaluation.

    Args:
        args: Parsed command line arguments.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config from file or build from args
    if hasattr(args, "config") and args.config:
        config = load_config_file(args.config)
        # Override with any CLI args if provided
        if args.model:
            config["model"]["path"] = args.model
        if args.num_samples:
            config["evaluation"]["num_samples"] = args.num_samples
    else:
        config = build_config_from_args(args)

    print(f"\n{'='*60}")
    print("Seriguela Quality Evaluation")
    print(f"{'='*60}")
    print(f"Model: {config['model']['path']}")
    print(f"Samples: {config['evaluation']['num_samples']}")
    print(f"Temperature: {config['generation']['temperature']}")
    print(f"{'='*60}\n")

    # Initialize components
    output_dir = getattr(args, "output_dir", "results/quality")
    storage = ResultStorage(base_dir=output_dir)

    # Create run
    run_id = storage.create_run(config)
    print(f"Run ID: {run_id}")
    print(f"Output: {storage.get_run_dir(run_id)}\n")

    # Load model
    print("Loading model...")
    start_time = time.time()

    loader = ModelLoader()
    model, tokenizer, base_model = loader.load(config["model"]["path"])

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Base model: {base_model}")
    print(f"Device: {loader.device}\n")

    # Setup generator
    gen_config = GenerationConfig(
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        top_k=config["generation"]["top_k"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        do_sample=config["generation"]["do_sample"],
    )
    generator = ExpressionGenerator(model, tokenizer, gen_config)

    # Setup prompt config
    prompt_cfg = config.get("prompt", {})
    prompt_config = PromptConfig(
        vars=prompt_cfg.get("vars", ["x_1"]),
        ops=prompt_cfg.get("ops", ["+", "-", "*", "/", "sin", "cos"]),
        cons=prompt_cfg.get("cons", "C"),
        format=prompt_cfg.get("format", "infix"),
    )

    # Build prompt
    prompt = generator.build_prompt(prompt_config)
    is_prefix = prompt_config.format == "prefix"

    print(f"Prompt format: {prompt_config.format}")
    print(f"Variables: {prompt_config.vars}")
    print(f"Operators: {prompt_config.ops}")
    print(f"Sample prompt: {prompt[:80]}...\n")

    # Setup extractor and validator
    extractor = ExpressionExtractor()
    validator = ExpressionValidator()
    calculator = MetricsCalculator()

    # Generate and evaluate
    print("Generating expressions...")
    num_samples = config["evaluation"]["num_samples"]
    results = []

    try:
        from tqdm import tqdm

        iterator = tqdm(range(num_samples), desc="Generating")
    except ImportError:
        iterator = range(num_samples)
        print(f"(Install tqdm for progress bar)")

    gen_start = time.time()

    for i in iterator:
        # Generate
        outputs = generator.generate(prompt)
        output = outputs[0] if outputs else ""

        # Extract expression
        extraction = extractor.extract(output, format_hint="json")
        expression = extraction.expression

        # Validate
        validation = validator.validate(expression or "", is_prefix=is_prefix)

        # Check constraints
        constraint_valid = True
        if prompt_config.vars:
            if validation.valid and not validation.variables_used.issubset(set(prompt_config.vars)):
                constraint_valid = False
        if prompt_config.ops:
            if validation.valid and not validation.operators_used.issubset(set(prompt_config.ops)):
                constraint_valid = False

        # Create result
        result = SampleResult(
            idx=i,
            prompt=prompt,
            output=output,
            expression=expression,
            validation=validation,
            constraint_valid=constraint_valid,
        )
        results.append(result)

        # Save incrementally
        storage.save_sample(run_id, result)

    gen_time = time.time() - gen_start

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculator.calculate(
        results,
        allowed_vars=prompt_config.vars,
        allowed_ops=prompt_config.ops,
    )

    # Save metrics and summary
    storage.save_metrics(run_id, metrics)
    storage.save_summary(run_id, metrics)

    # Print summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Total samples: {metrics.total_samples}")
    print(f"Valid: {metrics.valid_count} ({metrics.valid_rate:.1%})")
    print(f"Unique: {metrics.unique_count} ({metrics.diversity_rate:.1%})")
    print(f"Constraint adherence: {metrics.constraint_valid_count} ({metrics.constraint_adherence_rate:.1%})")
    print(f"\nGeneration time: {gen_time:.1f}s ({gen_time/num_samples:.2f}s/sample)")
    print(f"{'='*60}")
    print(f"\nResults saved to: {storage.get_run_dir(run_id)}")
    print(f"Run ID: {run_id}")

    # Show some example expressions
    print("\nSample expressions:")
    valid_results = [r for r in results if r.validation.valid][:5]
    for r in valid_results:
        print(f"  - {r.expression}")

    if metrics.error_types:
        print("\nCommon errors:")
        for error_type, count in sorted(metrics.error_types.items(), key=lambda x: -x[1])[:3]:
            print(f"  - {error_type}: {count}")

    # Auto-upload to HuggingFace if requested
    if getattr(args, "upload", False):
        print(f"\n{'='*60}")
        print("Uploading to HuggingFace...")
        print(f"{'='*60}")
        try:
            from core.hf_storage import HFResultStorage
            hf_repo = getattr(args, "hf_repo", "augustocsc/seriguela-results")
            hf_storage = HFResultStorage(repo_id=hf_repo)
            run_dir = storage.get_run_dir(run_id)
            success = hf_storage.upload_run(run_dir, eval_type="quality")
            if success:
                print(f"[OK] Uploaded to: https://huggingface.co/datasets/{hf_repo}/tree/main/quality/{run_id}")
            else:
                print("[FAIL] Upload failed - check HF_TOKEN environment variable")
        except Exception as e:
            print(f"[FAIL] Upload error: {e}")

    return run_id


def add_quality_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the quality command."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (HuggingFace repo or local path)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to generate (default: 500)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="Allowed variables, comma-separated (e.g., x_1,x_2,x_3)",
    )
    parser.add_argument(
        "--ops",
        type=str,
        help="Allowed operators, comma-separated (e.g., +,-,*,/,sin,cos)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/quality",
        help="Output directory for results (default: results/quality)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Automatically upload results to HuggingFace after evaluation",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="augustocsc/seriguela-results",
        help="HuggingFace repository for results (default: augustocsc/seriguela-results)",
    )


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="Quality evaluation")
    add_quality_arguments(parser)
    args = parser.parse_args()
    execute_quality(args)
