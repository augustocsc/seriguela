#!/usr/bin/env python3
"""
Quick factorial test - runs ONE config from each dimension to verify all work.

Tests:
- 5 algorithms (one each)
- 3 rewards (one each)
- 2 penalties (one each)
- 4 temperatures (one each)
- 3 prompts (one each)
- 4 noise levels (one each)

Total: 21 quick tests (not 1,440 factorial)
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

# Test configurations - one from each dimension
TEST_CONFIGS = [
    # Algorithm tests (vary algorithm, fix others)
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0},
    {"algo": "bon_grpo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0},
    {"algo": "pure_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0},
    {"algo": "pure_grpo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0},
    {"algo": "best_of_n", "reward": "length_penalized", "penalty": "gradient", "temp": "fixed_0.7", "prompt": "standard", "noise": 0.0},

    # Reward tests
    {"algo": "bon_ppo", "reward": "r2_clipped", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0},
    {"algo": "bon_ppo", "reward": "sr_ic", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0},

    # Penalty tests
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "binary", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0},

    # Temperature tests
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "fixed_0.7", "prompt": "standard", "noise": 0.0},
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "fixed_0.9", "prompt": "standard", "noise": 0.0},
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "linear_annealing", "prompt": "standard", "noise": 0.0},

    # Prompt tests
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "oracle", "noise": 0.0},
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "distractor", "noise": 0.0},

    # Noise tests
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.01},
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.05},
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.1},

    # Problem tests (N1, N5, N9)
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0, "problem": "nguyen_1"},
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0, "problem": "nguyen_5"},
    {"algo": "bon_ppo", "reward": "length_penalized", "penalty": "gradient", "temp": "cosine_annealing", "prompt": "standard", "noise": 0.0, "problem": "nguyen_9"},
]

MODEL = "augustocsc/gpt2_base_infix_682k"
DEFAULT_PROBLEM = "nguyen_5"


def build_command(config, max_steps=10, use_wandb=True):
    """Build the run_experiment.py command."""
    problem = config.get("problem", DEFAULT_PROBLEM)

    cmd = [
        "python", "run_experiment.py",
        "--algorithm", config["algo"],
        "--model", MODEL,
        "--problem", problem,
        "--reward", config["reward"],
        "--penalty", config["penalty"],
        "--temperature", config["temp"],
        "--prompt_type", config["prompt"],
        "--max_steps", str(max_steps),
        "--batch_size", "4",
        "--seeds", "42",
    ]

    if config["noise"] > 0:
        cmd.extend(["--noise_type", "gaussian", "--noise_level", str(config["noise"])])
    else:
        cmd.extend(["--noise_type", "none"])

    if use_wandb:
        cmd.append("--use_wandb")

    return cmd


def run_quick_test(max_steps=10, use_wandb=True):
    """Run quick test of all dimensions."""
    print("=" * 60)
    print(f"QUICK FACTORIAL TEST: {len(TEST_CONFIGS)} configurations")
    print(f"Max steps: {max_steps}, WandB: {use_wandb}")
    print("=" * 60)

    results = {
        "start_time": datetime.now().isoformat(),
        "max_steps": max_steps,
        "successes": [],
        "failures": [],
    }

    for i, config in enumerate(TEST_CONFIGS):
        config_name = f"{config['algo']}_{config['reward']}_{config['penalty']}_{config['temp']}_{config['prompt']}_noise{config['noise']}"
        if "problem" in config:
            config_name += f"_{config['problem']}"

        print(f"\n[{i+1}/{len(TEST_CONFIGS)}] Testing: {config_name}")

        cmd = build_command(config, max_steps, use_wandb)
        print(f"  Command: {' '.join(cmd[:10])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=Path(__file__).parent,
            )

            if result.returncode == 0:
                print(f"  ✓ SUCCESS")
                results["successes"].append(config_name)
            else:
                print(f"  ✗ FAILED")
                print(f"  Error: {result.stderr[-300:] if result.stderr else 'No stderr'}")
                results["failures"].append({"config": config_name, "error": result.stderr[-500:] if result.stderr else "No stderr"})

        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT")
            results["failures"].append({"config": config_name, "error": "Timeout"})
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results["failures"].append({"config": config_name, "error": str(e)})

    results["end_time"] = datetime.now().isoformat()

    # Save results
    output_file = Path(__file__).parent / f"quick_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(results['successes'])}/{len(TEST_CONFIGS)} passed")
    print(f"{'=' * 60}")

    if results["failures"]:
        print("\nFailed:")
        for f in results["failures"]:
            print(f"  - {f['config']}")

    return len(results["failures"]) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    success = run_quick_test(args.max_steps, not args.no_wandb)
    sys.exit(0 if success else 1)
