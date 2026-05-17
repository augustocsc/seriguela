#!/usr/bin/env python3
"""
Test all 1,440 factorial combinations with minimal steps.

Full factorial:
- 5 algorithms × 3 rewards × 2 penalties × 4 temperatures × 3 prompts × 4 noise levels
= 1,440 configurations

Run with --max_steps 3 to quickly verify all combinations work.
"""

import itertools
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

# All configuration options
ALGORITHMS = ["bon_ppo", "bon_grpo", "pure_ppo", "pure_grpo", "best_of_n"]
REWARDS = ["r2_clipped", "length_penalized", "sr_ic"]
PENALTIES = ["binary", "gradient"]
TEMPERATURES = ["fixed_0.7", "fixed_0.9", "linear_annealing", "cosine_annealing"]
PROMPTS = ["standard", "oracle", "distractor"]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]

# Models to test
MODELS = [
    "augustocsc/gpt2_base_infix_682k",
    "augustocsc/gpt2_base_prefix_682k",
    "augustocsc/gpt2_medium_infix_682k",
    "augustocsc/gpt2_medium_prefix_682k",
    "augustocsc/gpt2_large_infix_682k",
    "augustocsc/gpt2_large_prefix_682k",
]

# Representative problems (easy, hard, 2-variable)
PROBLEMS = ["nguyen_1", "nguyen_5", "nguyen_9"]


def generate_all_combinations():
    """Generate all factorial combinations."""
    combinations = list(itertools.product(
        ALGORITHMS,
        REWARDS,
        PENALTIES,
        TEMPERATURES,
        PROMPTS,
        NOISE_LEVELS,
    ))
    return combinations


def build_command(model, problem, algo, reward, penalty, temp, prompt, noise, max_steps=3):
    """Build the run_experiment.py command."""
    cmd = [
        "python", "run_experiment.py",
        "--algorithm", algo,
        "--model", model,
        "--problem", problem,
        "--reward", reward,
        "--penalty", penalty,
        "--temperature", temp,
        "--prompt_type", prompt,
        "--max_steps", str(max_steps),
        "--batch_size", "4",  # Small batch for quick test
        "--seeds", "42",
    ]

    if noise > 0:
        cmd.extend(["--noise_type", "gaussian", "--noise_level", str(noise)])
    else:
        cmd.extend(["--noise_type", "none"])

    return cmd


def run_test(max_steps=3, model_filter=None, problem_filter=None, save_results=True):
    """Run all combinations with minimal steps to verify they work."""
    combinations = generate_all_combinations()
    total_configs = len(combinations)

    models = [model_filter] if model_filter else MODELS[:1]  # Default: test 1 model
    problems = [problem_filter] if problem_filter else PROBLEMS[:1]  # Default: test 1 problem

    total_runs = len(models) * len(problems) * total_configs

    print(f"=" * 60)
    print(f"FACTORIAL TEST: {total_configs} configs × {len(models)} models × {len(problems)} problems = {total_runs} runs")
    print(f"Max steps per run: {max_steps}")
    print(f"=" * 60)

    results = {
        "start_time": datetime.now().isoformat(),
        "total_configs": total_configs,
        "models": models,
        "problems": problems,
        "max_steps": max_steps,
        "successes": [],
        "failures": [],
    }

    run_count = 0
    success_count = 0
    failure_count = 0

    for model in models:
        for problem in problems:
            for i, (algo, reward, penalty, temp, prompt, noise) in enumerate(combinations):
                run_count += 1
                config_name = f"{algo}_{reward}_{penalty}_{temp}_{prompt}_noise{noise}"

                print(f"\n[{run_count}/{total_runs}] Testing: {config_name}")
                print(f"  Model: {model}, Problem: {problem}")

                cmd = build_command(model, problem, algo, reward, penalty, temp, prompt, noise, max_steps)

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 min timeout per run
                        cwd=Path(__file__).parent,
                    )

                    if result.returncode == 0:
                        print(f"  ✓ SUCCESS")
                        success_count += 1
                        results["successes"].append({
                            "config": config_name,
                            "model": model,
                            "problem": problem,
                        })
                    else:
                        print(f"  ✗ FAILED (return code {result.returncode})")
                        print(f"  Error: {result.stderr[-500:] if result.stderr else 'No stderr'}")
                        failure_count += 1
                        results["failures"].append({
                            "config": config_name,
                            "model": model,
                            "problem": problem,
                            "error": result.stderr[-1000:] if result.stderr else "No stderr",
                            "returncode": result.returncode,
                        })

                except subprocess.TimeoutExpired:
                    print(f"  ✗ TIMEOUT")
                    failure_count += 1
                    results["failures"].append({
                        "config": config_name,
                        "model": model,
                        "problem": problem,
                        "error": "Timeout after 300s",
                    })
                except Exception as e:
                    print(f"  ✗ EXCEPTION: {e}")
                    failure_count += 1
                    results["failures"].append({
                        "config": config_name,
                        "model": model,
                        "problem": problem,
                        "error": str(e),
                    })

                # Progress summary
                print(f"  Progress: {success_count} success, {failure_count} failed")

    results["end_time"] = datetime.now().isoformat()
    results["success_count"] = success_count
    results["failure_count"] = failure_count

    # Save results
    if save_results:
        output_file = Path(__file__).parent / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total runs: {run_count}")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Success rate: {100 * success_count / run_count:.1f}%")

    if failure_count > 0:
        print(f"\nFailed configurations:")
        for f in results["failures"][:10]:  # Show first 10
            print(f"  - {f['config']}: {f['error'][:100]}")
        if len(results["failures"]) > 10:
            print(f"  ... and {len(results['failures']) - 10} more")

    return results


def print_factorial_info():
    """Print information about the factorial design."""
    combinations = generate_all_combinations()

    print("=" * 60)
    print("FACTORIAL EXPERIMENT DESIGN")
    print("=" * 60)
    print(f"\nDimensions:")
    print(f"  Algorithms:    {len(ALGORITHMS):3d} - {ALGORITHMS}")
    print(f"  Rewards:       {len(REWARDS):3d} - {REWARDS}")
    print(f"  Penalties:     {len(PENALTIES):3d} - {PENALTIES}")
    print(f"  Temperatures:  {len(TEMPERATURES):3d} - {TEMPERATURES}")
    print(f"  Prompts:       {len(PROMPTS):3d} - {PROMPTS}")
    print(f"  Noise levels:  {len(NOISE_LEVELS):3d} - {NOISE_LEVELS}")
    print(f"\nTotal configurations: {len(combinations)}")
    print(f"\nWith 6 models × 3 problems:")
    print(f"  Total runs: {len(combinations) * 6 * 3} = {len(combinations)} × 6 × 3")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test all factorial combinations")
    parser.add_argument("--info", action="store_true", help="Print factorial info and exit")
    parser.add_argument("--max_steps", type=int, default=3, help="Max steps per run (default: 3)")
    parser.add_argument("--model", type=str, default=None, help="Test specific model only")
    parser.add_argument("--problem", type=str, default=None, help="Test specific problem only")
    parser.add_argument("--all_models", action="store_true", help="Test all 6 models")
    parser.add_argument("--all_problems", action="store_true", help="Test all 3 problems")

    args = parser.parse_args()

    if args.info:
        print_factorial_info()
        sys.exit(0)

    # Determine what to test
    if args.all_models:
        model_filter = None  # Will use all models
    else:
        model_filter = args.model or MODELS[0]  # Default to first model

    if args.all_problems:
        problem_filter = None
    else:
        problem_filter = args.problem or PROBLEMS[0]

    # Run the test
    run_test(
        max_steps=args.max_steps,
        model_filter=model_filter if not args.all_models else None,
        problem_filter=problem_filter if not args.all_problems else None,
    )
