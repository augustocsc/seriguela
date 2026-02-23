#!/usr/bin/env python3
"""
Run remaining factorial experiments from a JSON file.

Usage:
    python run_remaining_experiment.py --model base_infix --problem nguyen_1 --max_steps 5000
"""

import subprocess
import sys
import json
import os
from datetime import datetime
from pathlib import Path
import argparse

# Models mapping
MODELS = {
    "base_infix": "augustocsc/gpt2_base_infix_682k",
    "base_prefix": "augustocsc/gpt2_base_prefix_682k",
    "medium_infix": "augustocsc/gpt2_medium_infix_682k",
    "medium_prefix": "augustocsc/gpt2_medium_prefix_682k",
    "large_infix": "augustocsc/gpt2_large_infix_682k",
    "large_prefix": "augustocsc/gpt2_large_prefix_682k",
}


def build_command(model_repo, problem, config, max_steps, batch_size, use_wandb=True, upload_hf=True):
    """Build run_experiment.py command from config tuple."""
    # config = (model, problem, algo, reward, penalty, temp, prompt, noise)
    algo = config[2]
    reward = config[3]
    penalty = config[4]
    temp = config[5]
    prompt = config[6]
    noise = config[7] if len(config) > 7 else 0.0

    cmd = [
        "python", "run_experiment.py",
        "--algorithm", algo,
        "--model", model_repo,
        "--problem", problem,
        "--reward", reward,
        "--penalty", penalty,
        "--temperature", temp,
        "--prompt_type", prompt,
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--seeds", "42",
    ]

    if noise > 0:
        cmd.extend(["--noise_type", "gaussian", "--noise_level", str(noise)])
    else:
        cmd.extend(["--noise_type", "none"])

    if use_wandb:
        cmd.append("--use_wandb")

    if upload_hf:
        cmd.append("--upload_hf")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run remaining factorial experiments")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()),
                        help="Model to run (e.g., base_infix)")
    parser.add_argument("--problem", type=str, required=True,
                        help="Problem to run (e.g., nguyen_1)")
    parser.add_argument("--remaining_file", type=str, default="remaining_base_configs.json",
                        help="JSON file with remaining configs")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Max steps per experiment")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--no_upload", action="store_true",
                        help="Disable HF upload")
    args = parser.parse_args()

    # Load remaining configs
    remaining_file = Path(args.remaining_file)
    if not remaining_file.exists():
        # Try to download from HF or use default path
        print(f"Error: {remaining_file} not found")
        sys.exit(1)

    with open(remaining_file) as f:
        all_remaining = json.load(f)

    # Filter for this model and problem
    configs = [c for c in all_remaining if c[0] == args.model and c[1] == args.problem]

    print(f"=" * 60)
    print(f"Running remaining experiments for {args.model} + {args.problem}")
    print(f"Total configs to run: {len(configs)}")
    print(f"=" * 60)

    if len(configs) == 0:
        print("No remaining configs for this model-problem combination!")
        return

    model_repo = MODELS[args.model]
    use_wandb = not args.no_wandb
    upload_hf = not args.no_upload

    # Adjust batch size for model size
    batch_size = args.batch_size
    if "large" in args.model:
        batch_size = min(batch_size, 16)
    elif "medium" in args.model:
        batch_size = min(batch_size, 24)

    completed = 0
    failed = 0

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config[2]} | {config[3]} | {config[4]} | {config[5]} | {config[6]} | noise={config[7]}")

        cmd = build_command(
            model_repo=model_repo,
            problem=args.problem,
            config=config,
            max_steps=args.max_steps,
            batch_size=batch_size,
            use_wandb=use_wandb,
            upload_hf=upload_hf,
        )

        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                completed += 1
                print(f"  ✓ Completed ({completed}/{i+1})")
            else:
                failed += 1
                print(f"  ✗ Failed (exit code {result.returncode})")
        except Exception as e:
            failed += 1
            print(f"  ✗ Exception: {e}")

    print(f"\n" + "=" * 60)
    print(f"FINISHED: {completed} completed, {failed} failed out of {len(configs)}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
