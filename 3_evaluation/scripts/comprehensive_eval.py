#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

Performs exhaustive testing across all models with various configurations:
- Temperature sweep
- Sampling method variations
- Different prompt configurations

Author: Augusto Cesar / Claude Code
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add paths
SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EVAL_DIR))


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    model: str
    temperature: float
    top_p: float
    top_k: int
    num_samples: int
    vars: List[str]
    ops: List[str]
    experiment_type: str  # temperature_sweep, prompt_variation, sampling_method


# All 6 models
MODELS = {
    "base_infix": "augustocsc/gpt2_base_infix_682k",
    "medium_infix": "augustocsc/gpt2_medium_infix_682k",
    "large_infix": "augustocsc/gpt2_large_infix_682k",
    "base_prefix": "augustocsc/gpt2_base_prefix_682k",
    "medium_prefix": "augustocsc/gpt2_medium_prefix_682k",
    "large_prefix": "augustocsc/gpt2_large_prefix_682k",
}

# Temperature configurations
TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]

# Sampling configurations
SAMPLING_CONFIGS = {
    "conservative": {"top_p": 0.8, "top_k": 30},
    "default": {"top_p": 0.9, "top_k": 50},
    "diverse": {"top_p": 0.95, "top_k": 100},
}

# Prompt configurations
PROMPT_CONFIGS = {
    "simple": {
        "vars": ["x_1"],
        "ops": ["+", "-", "*", "/"],
        "description": "Single variable, basic arithmetic"
    },
    "trigonometric": {
        "vars": ["x_1"],
        "ops": ["sin", "cos", "+", "-", "*"],
        "description": "Single variable with trigonometric functions"
    },
    "multivariate": {
        "vars": ["x_1", "x_2"],
        "ops": ["+", "-", "*", "/", "sin", "cos"],
        "description": "Two variables with mixed operations"
    },
    "complex": {
        "vars": ["x_1", "x_2", "x_3"],
        "ops": ["+", "-", "*", "/", "sin", "cos", "exp", "log"],
        "description": "Three variables with all operations"
    },
}


def generate_experiments(
    num_samples: int = 100,
    include_temperature: bool = True,
    include_prompts: bool = True,
    include_sampling: bool = True,
    models_filter: Optional[List[str]] = None,
) -> List[ExperimentConfig]:
    """Generate all experiment configurations."""

    experiments = []
    models_to_test = MODELS if not models_filter else {k: v for k, v in MODELS.items() if k in models_filter}

    # Default prompt and sampling for temperature sweep
    default_prompt = PROMPT_CONFIGS["simple"]
    default_sampling = SAMPLING_CONFIGS["default"]

    # Experiment 1: Temperature Sweep
    if include_temperature:
        for model_name, model_path in models_to_test.items():
            for temp in TEMPERATURES:
                exp = ExperimentConfig(
                    name=f"temp_{temp}_{model_name}",
                    model=model_path,
                    temperature=temp,
                    top_p=default_sampling["top_p"],
                    top_k=default_sampling["top_k"],
                    num_samples=num_samples,
                    vars=default_prompt["vars"],
                    ops=default_prompt["ops"],
                    experiment_type="temperature_sweep",
                )
                experiments.append(exp)

    # Experiment 2: Prompt Variations (at default temperature 0.7)
    if include_prompts:
        for model_name, model_path in models_to_test.items():
            for prompt_name, prompt_config in PROMPT_CONFIGS.items():
                exp = ExperimentConfig(
                    name=f"prompt_{prompt_name}_{model_name}",
                    model=model_path,
                    temperature=0.7,
                    top_p=default_sampling["top_p"],
                    top_k=default_sampling["top_k"],
                    num_samples=num_samples,
                    vars=prompt_config["vars"],
                    ops=prompt_config["ops"],
                    experiment_type="prompt_variation",
                )
                experiments.append(exp)

    # Experiment 3: Sampling Method Variations (at default temperature 0.7)
    if include_sampling:
        for model_name, model_path in models_to_test.items():
            for sampling_name, sampling_config in SAMPLING_CONFIGS.items():
                exp = ExperimentConfig(
                    name=f"sampling_{sampling_name}_{model_name}",
                    model=model_path,
                    temperature=0.7,
                    top_p=sampling_config["top_p"],
                    top_k=sampling_config["top_k"],
                    num_samples=num_samples,
                    vars=default_prompt["vars"],
                    ops=default_prompt["ops"],
                    experiment_type="sampling_method",
                )
                experiments.append(exp)

    return experiments


def run_experiment(exp: ExperimentConfig, output_dir: str, upload: bool = True) -> Dict:
    """Run a single experiment using the CLI."""

    print(f"\n{'='*70}")
    print(f"Running: {exp.name}")
    print(f"Model: {exp.model}")
    print(f"Temperature: {exp.temperature}, Top-p: {exp.top_p}, Top-k: {exp.top_k}")
    print(f"Variables: {exp.vars}, Operators: {exp.ops}")
    print(f"{'='*70}")

    # Build command
    cmd = [
        sys.executable, "-m", "3_evaluation.cli", "quality",
        "--model", exp.model,
        "--num-samples", str(exp.num_samples),
        "--temperature", str(exp.temperature),
        "--top-p", str(exp.top_p),
        "--top-k", str(exp.top_k),
        "--vars", ",".join(exp.vars),
        "--ops", ",".join(exp.ops),
        "--output-dir", output_dir,
    ]

    if upload:
        cmd.append("--upload")

    start_time = time.time()

    try:
        # Change to project root
        project_root = EVAL_DIR.parent
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes max per experiment
        )

        elapsed = time.time() - start_time

        return {
            "experiment": exp.name,
            "success": result.returncode == 0,
            "elapsed_seconds": elapsed,
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr": result.stderr[-500:] if result.stderr else None,
        }

    except subprocess.TimeoutExpired:
        return {
            "experiment": exp.name,
            "success": False,
            "error": "Timeout after 30 minutes",
        }
    except Exception as e:
        return {
            "experiment": exp.name,
            "success": False,
            "error": str(e),
        }


def run_all_experiments(
    experiments: List[ExperimentConfig],
    output_dir: str = "results/comprehensive",
    upload: bool = True,
    resume_from: Optional[str] = None,
) -> Dict:
    """Run all experiments and collect results."""

    results = []
    start_idx = 0

    # Resume from specific experiment if requested
    if resume_from:
        for i, exp in enumerate(experiments):
            if exp.name == resume_from:
                start_idx = i
                print(f"Resuming from experiment {i+1}: {resume_from}")
                break

    total = len(experiments)
    successful = 0
    failed = 0

    print(f"\n{'#'*70}")
    print(f"# COMPREHENSIVE MODEL EVALUATION")
    print(f"# Total experiments: {total}")
    print(f"# Starting from: {start_idx + 1}")
    print(f"# Output directory: {output_dir}")
    print(f"# Auto-upload: {upload}")
    print(f"{'#'*70}\n")

    overall_start = time.time()

    for i, exp in enumerate(experiments[start_idx:], start=start_idx + 1):
        print(f"\n[{i}/{total}] {exp.name}")

        result = run_experiment(exp, output_dir, upload)
        results.append(result)

        if result.get("success"):
            successful += 1
            print(f"[OK] Completed in {result.get('elapsed_seconds', 0):.1f}s")
        else:
            failed += 1
            print(f"[FAIL] {result.get('error', 'Unknown error')}")

        # Save progress after each experiment
        progress_file = Path(output_dir) / "progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, "w") as f:
            json.dump({
                "total": total,
                "completed": i,
                "successful": successful,
                "failed": failed,
                "last_experiment": exp.name,
                "results": results,
            }, f, indent=2)

    overall_elapsed = time.time() - overall_start

    # Final summary
    summary = {
        "total_experiments": total,
        "successful": successful,
        "failed": failed,
        "total_time_seconds": overall_elapsed,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    # Save final summary
    summary_file = Path(output_dir) / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# EVALUATION COMPLETE")
    print(f"# Successful: {successful}/{total}")
    print(f"# Failed: {failed}/{total}")
    print(f"# Total time: {overall_elapsed/60:.1f} minutes")
    print(f"# Summary saved: {summary_file}")
    print(f"{'#'*70}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation across all configurations"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per experiment (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comprehensive",
        help="Output directory for results (default: results/comprehensive)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Disable automatic upload to HuggingFace",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODELS.keys()),
        help="Specific models to test (default: all)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["temperature", "prompts", "sampling"],
        default=["temperature", "prompts", "sampling"],
        help="Types of experiments to run (default: all)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from a specific experiment name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments without running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Generate experiments
    experiments = generate_experiments(
        num_samples=args.num_samples,
        include_temperature="temperature" in args.experiments,
        include_prompts="prompts" in args.experiments,
        include_sampling="sampling" in args.experiments,
        models_filter=args.models,
    )

    if args.list:
        print(f"\nTotal experiments: {len(experiments)}\n")

        by_type = {}
        for exp in experiments:
            by_type.setdefault(exp.experiment_type, []).append(exp)

        for exp_type, exps in by_type.items():
            print(f"\n{exp_type.upper()} ({len(exps)} experiments):")
            for exp in exps:
                print(f"  - {exp.name}")

        return

    if args.dry_run:
        print(f"\nDry run - {len(experiments)} experiments would be executed:\n")
        for exp in experiments[:5]:
            print(f"  {exp.name}: temp={exp.temperature}, top_p={exp.top_p}, vars={exp.vars}")
        if len(experiments) > 5:
            print(f"  ... and {len(experiments) - 5} more")
        return

    # Run experiments
    run_all_experiments(
        experiments,
        output_dir=args.output_dir,
        upload=not args.no_upload,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
