#!/usr/bin/env python3
"""
Test 4: Temperature Schedule Comparison

Goal: Confirm cosine_annealing is superior to other schedules at 50 steps.
Also tests whether fixed_0.9 (high exploration) or linear_annealing provides
advantages on specific problem types.

Runs: 3 temperatures × 3 seeds = 9 experiments on nguyen_5 + nguyen_9 (50 steps each)
Key metric: Average R² per temperature across seeds and problems.
Estimated time: ~3 hours on Colab

Note: The CLI supports cosine_annealing (1.0→0.5), linear_annealing (1.0→0.5),
and fixed_0.9. Custom cosine ranges (e.g. 1.2→0.3) would require code changes
to expose t_max/t_min via CLI — deferred to Phase B if results warrant it.
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_RUNNER = "2_training/reinforcement/run_experiment.py"
OUTPUT_DIR = "results/pre_phase_b/test4_temp_compare"

BASE_CONFIG = {
    "algorithm": "bon_ppo",
    "model": "augustocsc/gpt2_base_infix_682k",
    "reward": "r2_clipped",
    "penalty": "gradient",
}

SEEDS = [42, 123, 456]
TEMPERATURES = ["cosine_annealing", "linear_annealing", "fixed_0.9"]
PROBLEMS = ["nguyen_5", "nguyen_9"]
MAX_STEPS = 50
BATCH_SIZE = 256

# ─── Helpers ───────────────────────────────────────────────────────────────────

def build_command(temperature: str, problem: str, seed: int) -> list:
    """Build the command to run a single experiment."""
    cmd = [
        sys.executable, EXPERIMENT_RUNNER,
        "--algorithm", BASE_CONFIG["algorithm"],
        "--model", BASE_CONFIG["model"],
        "--reward", BASE_CONFIG["reward"],
        "--penalty", BASE_CONFIG["penalty"],
        "--temperature", temperature,
        "--problem", problem,
        "--max_steps", str(MAX_STEPS),
        "--batch_size", str(BATCH_SIZE),
        "--patience", "999",
        "--seeds", str(seed),
        "--output_dir", OUTPUT_DIR,
        "--no_wandb",
    ]
    return cmd


def run_experiment(temperature: str, problem: str, seed: int, dry_run: bool = False) -> dict:
    """Run a single experiment and return results."""
    cmd = build_command(temperature, problem, seed)
    label = f"[Test4] {problem} | temp={temperature} | seed={seed}"

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"temperature": temperature, "problem": problem, "seed": seed,
                "best_r2": 0.0, "status": "dry_run"}

    result = subprocess.run(cmd, capture_output=False)

    # Find result file
    search_dir = Path(OUTPUT_DIR)
    jsons = sorted(search_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if jsons:
        latest = jsons[-1]
        with open(latest) as f:
            data = json.load(f)
        best_r2 = data.get("best_r2", -999.0)
        best_expr = data.get("best_expression", "N/A")
        print(f"  → best_r2={best_r2:.4f} | expr={best_expr}")
        return {"temperature": temperature, "problem": problem, "seed": seed,
                "best_r2": best_r2, "best_expression": best_expr, "file": str(latest)}

    print("  → WARNING: No result file found")
    return {"temperature": temperature, "problem": problem, "seed": seed,
            "best_r2": -999.0, "status": "no_result"}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test 4: Temperature Schedule Comparison")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    start = datetime.now()
    total_runs = len(TEMPERATURES) * len(PROBLEMS) * len(SEEDS)
    print(f"\n{'='*60}")
    print(f"  PRE-PHASE B — TEST 4: TEMPERATURE COMPARISON")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  Problems: {PROBLEMS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*60}")

    all_results = []
    for temp in TEMPERATURES:
        for problem in PROBLEMS:
            for seed in SEEDS:
                result = run_experiment(temp, problem, seed, dry_run=args.dry_run)
                all_results.append(result)

    # Print quick ranking
    if not args.dry_run and all_results:
        print(f"\n{'='*60}")
        print("  QUICK RANKING BY TEMPERATURE")
        print(f"{'='*60}")
        for temp in TEMPERATURES:
            temp_results = [r for r in all_results if r["temperature"] == temp and r["best_r2"] > -900]
            if temp_results:
                avg = sum(r["best_r2"] for r in temp_results) / len(temp_results)
                print(f"  {temp:<25} avg_r2={avg:.4f} (n={len(temp_results)})")

    # Save summary
    if not args.dry_run:
        summary_path = Path(OUTPUT_DIR) / "test4_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "test": "temperature_comparison",
                "config": BASE_CONFIG,
                "seeds": SEEDS,
                "temperatures": TEMPERATURES,
                "problems": PROBLEMS,
                "max_steps": MAX_STEPS,
                "results": all_results,
                "timestamp": start.isoformat(),
            }, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"  TEST 4 DONE in {elapsed:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
