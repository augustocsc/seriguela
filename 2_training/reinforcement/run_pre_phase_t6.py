#!/usr/bin/env python3
"""
Pre-Phase Test 6: Re-run bon_grpo and bon_ppo with the buffer fix.

This test reproduces the exact same parameters as Test 5 (pre_phase__t5)
but ONLY re-runs the bon variants (since pure variants are unaffected).

Results are saved to results/pre_phase__t6/ for side-by-side comparison.

Parameters (matching Test 5):
    - Model:       augustocsc/gpt2_base_infix_682k
    - Temperature:  cosine_annealing
    - Reward:       sr_ic
    - Penalty:      gradient
    - Max Steps:    50
    - Batch Size:   64
    - Algorithms:   bon_grpo, bon_ppo
    - Benchmarks:   nguyen_1, nguyen_5, nguyen_9
    - Seeds:        42, 123, 456

Total: 2 algorithms × 3 benchmarks × 3 seeds = 18 runs

Run on Google Colab (T4 GPU):
    cd 2_training/reinforcement
    python run_pre_phase_t6.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import subprocess
import time
import datetime
from pathlib import Path

# Configuration matching Test 5
MODEL = "augustocsc/gpt2_base_infix_682k"
ALGORITHMS = ["bon_grpo", "bon_ppo"]
BENCHMARKS = ["nguyen_1", "nguyen_5", "nguyen_9"]
SEEDS = [42, 123, 456]
TEMPERATURE = "cosine_annealing"
REWARD = "sr_ic"
PENALTY = "gradient"
MAX_STEPS = 50
BATCH_SIZE = 64
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"../../results/pre_phase__t6_{TIMESTAMP}"

# Working directory
SCRIPT_DIR = Path(__file__).parent


def run_experiment(algorithm, problem, seed):
    """Run a single experiment via run_experiment.py."""
    seeds_str = str(seed)

    cmd = [
        sys.executable, "run_experiment.py",
        "--algorithm", algorithm,
        "--model", MODEL,
        "--problem", problem,
        "--reward", REWARD,
        "--penalty", PENALTY,
        "--temperature", TEMPERATURE,
        "--max_steps", str(MAX_STEPS),
        "--batch_size", str(BATCH_SIZE),
        "--seeds", seeds_str,
        "--output_dir", OUTPUT_DIR,
        "--no_wandb",
        "--noise_type", "none",
        "--prompt_type", "standard",
    ]

    return cmd


def main():
    total_runs = len(ALGORITHMS) * len(BENCHMARKS) * len(SEEDS)
    completed = 0
    failed = 0
    failed_runs = []

    print("=" * 70)
    print("PRE-PHASE TEST 6: Buffer Re-tokenization Fix Comparison")
    print("=" * 70)
    print(f"Algorithms:  {ALGORITHMS}")
    print(f"Benchmarks:  {BENCHMARKS}")
    print(f"Seeds:       {SEEDS}")
    print(f"Total runs:  {total_runs}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print("=" * 70)

    start_time = time.time()

    for algo in ALGORITHMS:
        for problem in BENCHMARKS:
            for seed in SEEDS:
                run_id = f"{algo}_{problem}_seed{seed}"
                run_num = completed + failed + 1

                print(f"\n[{run_num}/{total_runs}] Running: {run_id}")
                print("-" * 50)

                cmd = run_experiment(algo, problem, seed)
                run_start = time.time()

                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(SCRIPT_DIR),
                        check=False,
                        capture_output=False,
                    )

                    elapsed = time.time() - run_start

                    if result.returncode == 0:
                        completed += 1
                        print(f"  ✓ Completed in {elapsed:.0f}s ({completed}/{total_runs})")
                    else:
                        failed += 1
                        failed_runs.append(run_id)
                        print(f"  ✗ Failed (exit code {result.returncode})")

                except Exception as e:
                    failed += 1
                    failed_runs.append(run_id)
                    print(f"  ✗ Exception: {e}")

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TEST 6 COMPLETE")
    print("=" * 70)
    print(f"Completed: {completed}/{total_runs}")
    print(f"Failed:    {failed}/{total_runs}")
    print(f"Total time: {total_time/60:.1f} minutes")

    if failed_runs:
        print(f"\nFailed runs:")
        for r in failed_runs:
            print(f"  - {r}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Compare with Test 5 results in: results/pre_phase__t5/")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PUSHING RESULTS TO GIT")
    print("=" * 70)
    try:
        # Assumes SCRIPT_DIR is inside the git repo
        repo_root = SCRIPT_DIR.parent.parent
        output_abs = Path(OUTPUT_DIR).resolve()
        
        # Pull first to avoid conflicts in Colab
        subprocess.run(["git", "pull", "--rebase"], cwd=repo_root, check=False)
        
        # Add the specific results folder
        subprocess.run(["git", "add", str(output_abs)], cwd=repo_root, check=True)
        
        # Commit
        subprocess.run(["git", "commit", "-m", f"chore: add Test 6 results for {TIMESTAMP}"], cwd=repo_root, check=True)
        
        # Push
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
        print(f"✅ Successfully pushed results from {OUTPUT_DIR} to git!")
    except Exception as e:
        print(f"❌ Failed to push to git: {e}")
        print("You may need to push manually or check git credentials in Colab.")


if __name__ == "__main__":
    main()
