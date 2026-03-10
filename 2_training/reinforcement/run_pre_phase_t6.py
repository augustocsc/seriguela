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
import shutil
from pathlib import Path

# Configuration matching Test 5
MODEL = "augustocsc/gpt2_base_infix_682k"
ALGORITHMS = ["bon_grpo", "bon_ppo"]
BENCHMARKS = ["nguyen_1", "nguyen_5", "nguyen_9"]
SEEDS = [42, 123, 456]
TEMPERATURE = "cosine_annealing"
REWARD = "sr_ic"
PENALTY = "gradient"
MAX_STEPS = 500
BATCH_SIZE = 1024
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
    print("SAVING RESULTS TO GOOGLE DRIVE")
    print("=" * 70)
    
    # Target directory on Google Drive
    drive_base = Path("/content/drive/MyDrive/seriguela_results")
    drive_target = drive_base / Path(OUTPUT_DIR).name
    
    try:
        # Check if drive is mounted
        if not Path("/content/drive").exists():
            print("⚠️ Google Drive not found at /content/drive.")
            print("Did you forget to mount it? Run this in Colab:")
            print("from google.colab import drive")
            print("drive.mount('/content/drive')")
            print("\nAfter mounting, you can manually copy with:")
            print(f"!cp -r {OUTPUT_DIR} /content/drive/MyDrive/seriguela_results/")
        else:
            print(f"Copying {OUTPUT_DIR} -> {drive_target}")
            
            # Create base dir if it doesn't exist
            drive_base.mkdir(parents=True, exist_ok=True)
            
            # Copy everything
            shutil.copytree(OUTPUT_DIR, drive_target, dirs_exist_ok=True)
            
            print(f"✅ Successfully saved results to Google Drive!")
            
    except Exception as e:
        print(f"❌ Failed to copy to Google Drive: {e}")
        print("You may need to copy manually.")


if __name__ == "__main__":
    main()
