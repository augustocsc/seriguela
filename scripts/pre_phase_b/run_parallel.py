#!/usr/bin/env python3
"""
Parallel Experiment Launcher for Pre-Phase B Tests

Runs multiple experiments simultaneously to maximize T4 GPU utilization.
Results are saved to --output_dir as each experiment completes, so a
Colab session reset only loses in-flight experiments.

Usage (Google Colab):
    python scripts/pre_phase_b/run_parallel.py \\
        --test test2 \\
        --output_dir /content/drive/MyDrive/seriguela_results/pre_phase_b \\
        --max_parallel 4 \\
        --batch_size 2048

Usage (local):
    python scripts/pre_phase_b/run_parallel.py --test test4 --max_parallel 2

Memory guide (T4 = 15GB):
    --max_parallel 2   ≈  4GB GPU RAM  (safe)
    --max_parallel 4   ≈  8GB GPU RAM  (recommended)
    --max_parallel 6   ≈ 12GB GPU RAM  (aggressive)
"""

import subprocess
import sys
import json
import argparse
import time
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

EXPERIMENT_RUNNER = "2_training/reinforcement/run_experiment.py"

# Winner config (fixed across all tests)
BASE_CONFIG = {
    "algorithm": "bon_ppo",
    "model": "augustocsc/gpt2_base_infix_682k",
    "reward": "r2_clipped",
    "penalty": "gradient",
}

# ─── Experiment Matrices ───────────────────────────────────────────────────────

def get_test1_matrix():
    """Test 1: Multi-seed robustness — 3 seeds × 3 problems (9 runs)."""
    jobs = []
    for problem in ["nguyen_1", "nguyen_5", "nguyen_9"]:
        for seed in [42, 123, 456]:
            jobs.append({
                "label": f"test1 | {problem} | cosine_annealing | seed={seed}",
                "problem": problem,
                "temperature": "cosine_annealing",
                "seed": seed,
                "max_steps": 50,
                "patience": 999,
            })
    return jobs


def get_test2_matrix():
    """Test 2: Nguyen-5 failure debug — 2 temps × 3 seeds (6 runs)."""
    jobs = []
    for temp in ["cosine_annealing", "fixed_0.9"]:
        for seed in [42, 123, 456]:
            jobs.append({
                "label": f"test2 | nguyen_5 | {temp} | seed={seed}",
                "problem": "nguyen_5",
                "temperature": temp,
                "seed": seed,
                "max_steps": 50,
                "patience": 999,
            })
    return jobs


def get_test3_matrix():
    """Test 3: Convergence profile — 3 problems × 200 steps (3 runs)."""
    jobs = []
    for problem in ["nguyen_1", "nguyen_5", "nguyen_9"]:
        jobs.append({
            "label": f"test3 | {problem} | cosine_annealing | seed=42 | 200 steps",
            "problem": problem,
            "temperature": "cosine_annealing",
            "seed": 42,
            "max_steps": 200,
            "patience": 999,  # No early stopping — want full learning curve
        })
    return jobs


def get_test4_matrix():
    """Test 4: Temperature comparison — 3 temps × 2 problems × 3 seeds (18 runs)."""
    jobs = []
    for temp in ["cosine_annealing", "linear_annealing", "fixed_0.9"]:
        for problem in ["nguyen_5", "nguyen_9"]:
            for seed in [42, 123, 456]:
                jobs.append({
                    "label": f"test4 | {problem} | {temp} | seed={seed}",
                    "problem": problem,
                    "temperature": temp,
                    "seed": seed,
                    "max_steps": 50,
                    "patience": 999,
                })
    return jobs


def get_test5_matrix():
    """Test 5: RL algorithm comparison — 4 algorithms × 3 problems × 3 seeds (36 runs)."""
    jobs = []
    for algo in ["bon_ppo", "bon_grpo", "pure_ppo", "pure_grpo"]:
        for problem in ["nguyen_1", "nguyen_5", "nguyen_9"]:
            for seed in [42, 123, 456]:
                jobs.append({
                    "label": f"test5 | {algo} | {problem} | seed={seed}",
                    "algorithm": algo,
                    "problem": problem,
                    "temperature": "cosine_annealing",
                    "seed": seed,
                    "max_steps": 50,
                    "patience": 999,
                })
    return jobs


def get_all_matrix():
    """All tests combined."""
    return get_test1_matrix() + get_test2_matrix() + get_test3_matrix() + get_test4_matrix() + get_test5_matrix()


# ─── Persistence: Skip Already Done ──────────────────────────────────────────

def is_already_done(job: dict, output_dir: str) -> bool:
    """Check if a result file already exists for this job (session-crash resilience)."""
    search_dir = Path(output_dir)
    if not search_dir.exists():
        return False

    # Look for any JSON containing matching config metadata
    for json_file in search_dir.rglob("aggregate_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if (data.get("problem") == job["problem"] and
                    str(job["seed"]) in str(data.get("seeds", []))):
                return True
        except Exception:
            pass

    algo = job.get("algorithm", BASE_CONFIG["algorithm"])
    temp_map = {
        "cosine_annealing": "cosine_1.0_0.5",
        "linear_annealing": "linear_1.0_0.5",
        "fixed_0.9": "fixed_0.9",
        "fixed_0.7": "fixed_0.7"
    }
    job_temp = temp_map.get(job["temperature"], job["temperature"])

    # Also check individual result files
    for json_file in search_dir.rglob("results_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            if (data.get("problem") == job["problem"] and
                data.get("seed") == job["seed"] and
                data.get("algorithm", BASE_CONFIG["algorithm"]) == algo and
                data.get("temp_scheduler") == job_temp):
                return True
        except Exception:
            pass

    return False


# ─── Command Builder ───────────────────────────────────────────────────────────

def build_command(job: dict, output_dir: str, batch_size: int) -> List[str]:
    """Build run_experiment.py command for a single job."""
    return [
        sys.executable, EXPERIMENT_RUNNER,
        "--algorithm", job.get("algorithm", BASE_CONFIG["algorithm"]),
        "--model", BASE_CONFIG["model"],
        "--reward", BASE_CONFIG["reward"],
        "--penalty", BASE_CONFIG["penalty"],
        "--temperature", job["temperature"],
        "--problem", job["problem"],
        "--max_steps", str(job["max_steps"]),
        "--batch_size", str(batch_size),
        "--patience", str(job["patience"]),
        "--seeds", str(job["seed"]),
        "--output_dir", output_dir,
        "--no_wandb",
    ]


# ─── Single Job Runner (runs in a thread) ─────────────────────────────────────

def run_job(job: dict, output_dir: str, batch_size: int,
            dry_run: bool = False, job_idx: int = 0, total: int = 0) -> dict:
    """Execute a single experiment subprocess."""
    label = job["label"]
    cmd = build_command(job, output_dir, batch_size)

    prefix = f"[{job_idx+1}/{total}]"
    print(f"\n{prefix} START  {label}", flush=True)

    if dry_run:
        print(f"{prefix} CMD    {' '.join(cmd)}", flush=True)
        return {**job, "status": "dry_run", "best_r2": 0.0}

    t0 = time.time()
    try:
        # Redirect output to avoid interleaved mess from parallel processes
        log_file = Path(output_dir) / f"log_{job['problem']}_{job['temperature']}_seed{job['seed']}.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"error (code {proc.returncode})"
        print(f"{prefix} DONE   {label}  ({elapsed/60:.1f} min)  [{status}]", flush=True)

        return {**job, "status": status, "elapsed_min": elapsed / 60, "log": str(log_file)}

    except Exception as e:
        elapsed = time.time() - t0
        print(f"{prefix} FAIL   {label}  ({elapsed/60:.1f} min)  [{e}]", flush=True)
        return {**job, "status": f"exception: {e}", "elapsed_min": elapsed / 60}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel experiment launcher — maximizes T4 GPU utilization"
    )
    parser.add_argument("--test", choices=["test1", "test2", "test3", "test4", "test5", "all"], default="all",
                        help="Which test to run (default: all)")
    parser.add_argument("--output_dir", type=str,
                        default="results/pre_phase_b",
                        help="Output directory (use Drive path on Colab: /content/drive/MyDrive/...)")
    parser.add_argument("--max_parallel", type=int, default=4,
                        help="Max simultaneous experiments (default: 4, uses ~8GB on T4)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size per experiment (default: 2048 for T4)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_done", action="store_true", default=True,
                        help="Skip experiments that already have result files (default: True)")
    args = parser.parse_args()

    # Select job matrix
    if args.test == "test1":
        all_jobs = get_test1_matrix()
    elif args.test == "test2":
        all_jobs = get_test2_matrix()
    elif args.test == "test3":
        all_jobs = get_test3_matrix()
    elif args.test == "test4":
        all_jobs = get_test4_matrix()
    elif args.test == "test5":
        all_jobs = get_test5_matrix()
    else:
        all_jobs = get_all_matrix()

    # Filter already-done jobs (crash resilience)
    if args.skip_done and not args.dry_run:
        pending = [j for j in all_jobs if not is_already_done(j, args.output_dir)]
        skipped = len(all_jobs) - len(pending)
        if skipped:
            print(f"\n  ✅ Skipping {skipped} already-completed experiments")
    else:
        pending = all_jobs

    total = len(pending)
    start = datetime.now()

    print(f"\n{'='*65}")
    print(f"  PARALLEL LAUNCHER — {args.test.upper()}")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Jobs: {total} | Parallel: {args.max_parallel} | Batch: {args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*65}")

    if total == 0:
        print("  All experiments already done! Run analyze_results.py to see results.")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run with thread pool (each thread spawns one subprocess)
    completed_results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(run_job, job, args.output_dir, args.batch_size,
                            args.dry_run, i, total): job
            for i, job in enumerate(pending)
        }
        for future in as_completed(futures):
            result = future.result()
            completed_results.append(result)

    # Summary
    elapsed = (datetime.now() - start).total_seconds() / 60
    ok = sum(1 for r in completed_results if r.get("status") == "ok")
    failed = sum(1 for r in completed_results if r.get("status", "").startswith("error"))

    print(f"\n{'='*65}")
    print(f"  DONE in {elapsed:.1f} minutes")
    print(f"  ✅ Succeeded: {ok}/{total}   ❌ Failed: {failed}/{total}")
    print(f"{'='*65}")
    print(f"\n  Now run: python scripts/pre_phase_b/analyze_results.py")
    print(f"  With:    --results_dir {args.output_dir}\n")


if __name__ == "__main__":
    main()
