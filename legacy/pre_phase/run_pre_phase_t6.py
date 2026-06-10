#!/usr/bin/env python3
"""
Pre-Phase Test 6: Re-run bon_grpo and bon_ppo with the buffer fix.

Features:
    - Skips experiments whose aggregate JSON already exists on Google Drive
    - Runs multiple experiments in parallel (configurable PARALLEL_WORKERS)
    - Auto-saves results to Google Drive after each experiment completes

Parameters (matching Test 5):
    - Model:       augustocsc/gpt2_base_infix_682k
    - Temperature:  cosine_annealing
    - Reward:       sr_ic
    - Penalty:      gradient
    - Max Steps:    500
    - Batch Size:   1024
    - Algorithms:   bon_grpo, bon_ppo
    - Benchmarks:   nguyen_1, nguyen_5, nguyen_9
    - Seeds:        42, 123, 456

Total: 2 algorithms x 3 benchmarks x 3 seeds = 18 runs

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
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Configuration matching Test 5 ───────────────────────────────────
MODEL = "augustocsc/gpt2_base_infix_682k"
ALGORITHMS = ["bon_grpo", "bon_ppo"]
BENCHMARKS = ["nguyen_1", "nguyen_5", "nguyen_9"]
SEEDS = [42, 123, 456]
TEMPERATURE = "cosine_annealing"
REWARD = "sr_ic"
PENALTY = "gradient"
MAX_STEPS = 500
BATCH_SIZE = 1024

# ─── Parallelism ─────────────────────────────────────────────────────
PARALLEL_WORKERS = 2  # Number of experiments to run simultaneously

# ─── Paths ───────────────────────────────────────────────────────────
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"../../results/pre_phase__t6_{TIMESTAMP}"
SCRIPT_DIR = Path(__file__).parent
DRIVE_BASE = Path("/content/drive/MyDrive/seriguela_results")


def find_completed_runs():
    """Scan all pre_phase__t6_* folders on Google Drive for completed aggregate JSONs."""
    completed = set()

    # Check Google Drive folders
    if DRIVE_BASE.exists():
        for folder in DRIVE_BASE.glob("pre_phase__t6_*"):
            if folder.is_dir():
                for json_file in folder.glob("aggregate_*.json"):
                    # Extract run_id from filename: aggregate_bon_grpo_nguyen_1_seed42.json
                    name = json_file.stem  # aggregate_bon_grpo_nguyen_1_seed42
                    run_id = name.replace("aggregate_", "")  # bon_grpo_nguyen_1_seed42
                    completed.add(run_id)

    # Also check local results folders
    local_results = Path(SCRIPT_DIR) / ".." / ".." / "results"
    if local_results.exists():
        for folder in local_results.glob("pre_phase__t6_*"):
            if folder.is_dir():
                for json_file in folder.glob("aggregate_*.json"):
                    name = json_file.stem
                    run_id = name.replace("aggregate_", "")
                    completed.add(run_id)

    return completed


def copy_to_drive():
    """Copies current results to Google Drive."""
    drive_target = DRIVE_BASE / Path(OUTPUT_DIR).name

    try:
        if not Path("/content/drive").exists():
            return False

        DRIVE_BASE.mkdir(parents=True, exist_ok=True)
        
        # Only copy if output dir exists and has files
        out_path = Path(SCRIPT_DIR) / OUTPUT_DIR
        if not out_path.exists():
            # try absolute
            out_path = Path(OUTPUT_DIR)
        if not out_path.exists():
            return False
            
        shutil.copytree(str(out_path), str(drive_target), dirs_exist_ok=True)
        print(f"  [Auto-Save] ✅ Saved to {drive_target}")
        return True
    except Exception as e:
        print(f"  [Auto-Save] ❌ Failed: {e}")
        return False


def run_single_experiment(algo, problem, seed):
    """Run a single experiment. Returns (run_id, success, elapsed)."""
    run_id = f"{algo}_{problem}_seed{seed}"
    seeds_str = str(seed)

    cmd = [
        sys.executable, "run_experiment.py",
        "--algorithm", algo,
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

    run_start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            check=False,
            capture_output=False,
        )
        elapsed = time.time() - run_start
        success = result.returncode == 0
        return run_id, success, elapsed

    except Exception as e:
        elapsed = time.time() - run_start
        print(f"  ✗ [{run_id}] Exception: {e}")
        return run_id, False, elapsed


def main():
    # ─── Build list of all runs ──────────────────────────────────────
    all_runs = []
    for algo in ALGORITHMS:
        for problem in BENCHMARKS:
            for seed in SEEDS:
                all_runs.append((algo, problem, seed))

    total_runs = len(all_runs)

    # ─── Check which are already completed ───────────────────────────
    already_done = find_completed_runs()

    pending_runs = []
    skipped = 0
    for algo, problem, seed in all_runs:
        run_id = f"{algo}_{problem}_seed{seed}"
        if run_id in already_done:
            skipped += 1
        else:
            pending_runs.append((algo, problem, seed))

    # ─── Print banner ────────────────────────────────────────────────
    print("=" * 70)
    print("PRE-PHASE TEST 6: Buffer Re-tokenization Fix Comparison")
    print("=" * 70)
    print(f"Algorithms:       {ALGORITHMS}")
    print(f"Benchmarks:       {BENCHMARKS}")
    print(f"Seeds:            {SEEDS}")
    print(f"Total runs:       {total_runs}")
    print(f"Already complete: {skipped}")
    print(f"Pending runs:     {len(pending_runs)}")
    print(f"Parallel workers: {PARALLEL_WORKERS}")
    print(f"Output dir:       {OUTPUT_DIR}")
    print("=" * 70)

    if already_done:
        print(f"\n⏭️  Skipping {skipped} already-completed runs:")
        for run_id in sorted(already_done):
            print(f"    ✓ {run_id}")

    if not pending_runs:
        print("\n🎉 All runs are already complete! Nothing to do.")
        return

    print(f"\n🚀 Launching {len(pending_runs)} experiments ({PARALLEL_WORKERS} at a time)...\n")

    # ─── Run experiments in parallel ─────────────────────────────────
    completed = 0
    failed = 0
    failed_runs = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # Submit all pending runs
        future_to_run = {}
        for algo, problem, seed in pending_runs:
            future = executor.submit(run_single_experiment, algo, problem, seed)
            future_to_run[future] = f"{algo}_{problem}_seed{seed}"

        # Collect results as they finish
        for future in as_completed(future_to_run):
            run_id = future_to_run[future]
            try:
                rid, success, elapsed = future.result()
                if success:
                    completed += 1
                    print(f"\n  ✓ [{rid}] Completed in {elapsed:.0f}s  ({completed}/{len(pending_runs)} done)")
                    copy_to_drive()
                else:
                    failed += 1
                    failed_runs.append(rid)
                    print(f"\n  ✗ [{rid}] Failed after {elapsed:.0f}s")
                    copy_to_drive()
            except Exception as e:
                failed += 1
                failed_runs.append(run_id)
                print(f"\n  ✗ [{run_id}] Exception: {e}")
                copy_to_drive()

    total_time = time.time() - start_time

    # ─── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 6 COMPLETE")
    print("=" * 70)
    print(f"Skipped (already done): {skipped}")
    print(f"Completed this run:     {completed}/{len(pending_runs)}")
    print(f"Failed this run:        {failed}/{len(pending_runs)}")
    print(f"Total time:             {total_time/60:.1f} minutes")

    if failed_runs:
        print(f"\nFailed runs:")
        for r in failed_runs:
            print(f"  - {r}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Compare with Test 5 results in: results/pre_phase__t5/")
    print("=" * 70)

    # Final copy
    copy_to_drive()


if __name__ == "__main__":
    main()
