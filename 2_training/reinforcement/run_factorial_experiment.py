#!/usr/bin/env python3
"""
Run full factorial experiment for symbolic regression.

Full factorial design:
- 6 models × 3 problems × 1,440 configs = 25,920 runs

Each instance runs a subset of the full experiment.
Usage:
    python run_factorial_experiment.py --model_idx 0 --max_steps 5000
    python run_factorial_experiment.py --model_idx 0 --problem nguyen_5 --max_steps 5000
"""

import itertools
import subprocess
import sys
import json
import os
from datetime import datetime
from pathlib import Path
import argparse

# All configuration options
ALGORITHMS = ["bon_ppo", "bon_grpo", "pure_ppo", "pure_grpo", "best_of_n"]
REWARDS = ["r2_clipped", "length_penalized", "sr_ic"]
PENALTIES = ["binary", "gradient"]
TEMPERATURES = ["fixed_0.7", "fixed_0.9", "linear_annealing", "cosine_annealing"]
PROMPTS = ["standard", "oracle", "distractor"]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]

# Models
MODELS = [
    ("augustocsc/gpt2_base_infix_682k", "base_infix"),
    ("augustocsc/gpt2_base_prefix_682k", "base_prefix"),
    ("augustocsc/gpt2_medium_infix_682k", "medium_infix"),
    ("augustocsc/gpt2_medium_prefix_682k", "medium_prefix"),
    ("augustocsc/gpt2_large_infix_682k", "large_infix"),
    ("augustocsc/gpt2_large_prefix_682k", "large_prefix"),
]

# Problems (representative subset)
PROBLEMS = ["nguyen_1", "nguyen_5", "nguyen_9"]


def generate_all_configs():
    """Generate all 1,440 factorial configurations."""
    configs = []
    for algo, reward, penalty, temp, prompt, noise in itertools.product(
        ALGORITHMS, REWARDS, PENALTIES, TEMPERATURES, PROMPTS, NOISE_LEVELS
    ):
        configs.append({
            "algorithm": algo,
            "reward": reward,
            "penalty": penalty,
            "temperature": temp,
            "prompt": prompt,
            "noise": noise,
        })
    return configs


def build_command(model_repo, problem, config, max_steps, batch_size, use_wandb=True, upload_hf=True):
    """Build run_experiment.py command."""
    cmd = [
        "python", "run_experiment.py",
        "--algorithm", config["algorithm"],
        "--model", model_repo,
        "--problem", problem,
        "--reward", config["reward"],
        "--penalty", config["penalty"],
        "--temperature", config["temperature"],
        "--prompt_type", config["prompt"],
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--seeds", "42",
    ]

    if config["noise"] > 0:
        cmd.extend(["--noise_type", "gaussian", "--noise_level", str(config["noise"])])
    else:
        cmd.extend(["--noise_type", "none"])

    if use_wandb:
        cmd.append("--use_wandb")

    if upload_hf:
        cmd.append("--upload_hf")

    return cmd


def run_experiment(
    model_idx: int = None,
    problem: str = None,
    max_steps: int = 5000,
    batch_size: int = 32,
    use_wandb: bool = True,
    start_config: int = 0,
    end_config: int = None,
):
    """Run factorial experiment for specified model/problem subset."""

    configs = generate_all_configs()
    total_configs = len(configs)

    # Determine which models to run
    if model_idx is not None:
        models = [MODELS[model_idx]]
    else:
        models = MODELS

    # Determine which problems to run
    if problem is not None:
        problems = [problem]
    else:
        problems = PROBLEMS

    # Slice configs if specified
    if end_config is None:
        end_config = total_configs
    configs = configs[start_config:end_config]

    total_runs = len(models) * len(problems) * len(configs)

    print("=" * 70)
    print(f"FACTORIAL EXPERIMENT")
    print("=" * 70)
    print(f"Models: {[m[1] for m in models]}")
    print(f"Problems: {problems}")
    print(f"Configs: {len(configs)} (indices {start_config}-{end_config})")
    print(f"Total runs: {total_runs}")
    print(f"Max steps: {max_steps}")
    print(f"WandB: {use_wandb}")
    print("=" * 70)

    results = {
        "start_time": datetime.now().isoformat(),
        "models": [m[1] for m in models],
        "problems": problems,
        "config_range": [start_config, end_config],
        "total_runs": total_runs,
        "max_steps": max_steps,
        "successes": [],
        "failures": [],
    }

    run_count = 0
    success_count = 0
    failure_count = 0

    for model_repo, model_name in models:
        for prob in problems:
            for config_idx, config in enumerate(configs):
                run_count += 1
                config_name = f"{config['algorithm']}_{config['reward']}_{config['penalty']}_{config['temperature']}_{config['prompt']}_noise{config['noise']}"

                print(f"\n[{run_count}/{total_runs}] {model_name} / {prob}")
                print(f"  Config: {config_name}")

                cmd = build_command(model_repo, prob, config, max_steps, batch_size, use_wandb, upload_hf=True)

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=3600,  # 1 hour timeout
                        cwd=Path(__file__).parent,
                    )

                    if result.returncode == 0:
                        print(f"  ✓ SUCCESS")
                        success_count += 1
                        results["successes"].append({
                            "model": model_name,
                            "problem": prob,
                            "config": config_name,
                        })
                    else:
                        print(f"  ✗ FAILED")
                        error_msg = result.stderr[-500:] if result.stderr else "No stderr"
                        print(f"  Error: {error_msg[:200]}")
                        failure_count += 1
                        results["failures"].append({
                            "model": model_name,
                            "problem": prob,
                            "config": config_name,
                            "error": error_msg,
                        })

                except subprocess.TimeoutExpired:
                    print(f"  ✗ TIMEOUT (1h)")
                    failure_count += 1
                    results["failures"].append({
                        "model": model_name,
                        "problem": prob,
                        "config": config_name,
                        "error": "Timeout after 1 hour",
                    })
                except Exception as e:
                    print(f"  ✗ EXCEPTION: {e}")
                    failure_count += 1
                    results["failures"].append({
                        "model": model_name,
                        "problem": prob,
                        "config": config_name,
                        "error": str(e),
                    })

                # Progress
                print(f"  Progress: {success_count}/{run_count} success ({100*success_count/run_count:.1f}%)")

    results["end_time"] = datetime.now().isoformat()
    results["success_count"] = success_count
    results["failure_count"] = failure_count

    # Save results
    output_file = Path(__file__).parent / f"factorial_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total runs: {run_count}")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Success rate: {100*success_count/run_count:.1f}%")

    return results


def print_info():
    """Print experiment info."""
    configs = generate_all_configs()

    print("=" * 70)
    print("FULL FACTORIAL EXPERIMENT DESIGN")
    print("=" * 70)
    print(f"\nDimensions:")
    print(f"  Algorithms:    {len(ALGORITHMS):4d} - {ALGORITHMS}")
    print(f"  Rewards:       {len(REWARDS):4d} - {REWARDS}")
    print(f"  Penalties:     {len(PENALTIES):4d} - {PENALTIES}")
    print(f"  Temperatures:  {len(TEMPERATURES):4d} - {TEMPERATURES}")
    print(f"  Prompts:       {len(PROMPTS):4d} - {PROMPTS}")
    print(f"  Noise levels:  {len(NOISE_LEVELS):4d} - {NOISE_LEVELS}")
    print(f"\nTotal configs: {len(configs)}")
    print(f"\nModels: {len(MODELS)}")
    for repo, name in MODELS:
        print(f"  [{MODELS.index((repo, name))}] {name}: {repo}")
    print(f"\nProblems: {len(PROBLEMS)} - {PROBLEMS}")
    print(f"\nFull experiment: {len(MODELS)} × {len(PROBLEMS)} × {len(configs)} = {len(MODELS) * len(PROBLEMS) * len(configs)} runs")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run factorial experiment")
    parser.add_argument("--info", action="store_true", help="Print experiment info")
    parser.add_argument("--model_idx", type=int, default=None,
                        help="Model index (0-5). If not set, runs all models")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem to run (nguyen_1, nguyen_5, nguyen_9). If not set, runs all")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--start_config", type=int, default=0,
                        help="Start config index (for parallelization)")
    parser.add_argument("--end_config", type=int, default=None,
                        help="End config index (for parallelization)")

    args = parser.parse_args()

    if args.info:
        print_info()
        sys.exit(0)

    run_experiment(
        model_idx=args.model_idx,
        problem=args.problem,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
        start_config=args.start_config,
        end_config=args.end_config,
    )
