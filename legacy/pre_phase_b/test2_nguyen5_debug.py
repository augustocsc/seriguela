#!/usr/bin/env python3
"""
Test 2: Nguyen-5 Failure Investigation

Goal: Investigate why Nguyen-5 scored R²≈0.0 at 50 steps in Stage 3 validation
despite scoring R²=1.0 in Stage 2 (10 steps). Test multiple seeds and temperatures.

Runs: 3 seeds × 2 temperatures = 6 experiments (50 steps each)
Key metric: Whether any seed/temp combo reproduces R²≥0.5 on Nguyen-5.
Estimated time: ~2 hours on Colab
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_RUNNER = "2_training/reinforcement/run_experiment.py"
DEFAULT_OUTPUT_DIR = "results/pre_phase_b/test2_nguyen5_debug"

BASE_CONFIG = {
    "algorithm": "bon_ppo",
    "model": "augustocsc/gpt2_base_infix_682k",
    "reward": "r2_clipped",
    "penalty": "gradient",
}

SEEDS = [42, 123, 456]
TEMPERATURES = ["cosine_annealing", "fixed_0.9"]
PROBLEM = "nguyen_5"
MAX_STEPS = 50
BATCH_SIZE = 512  # T4 (16GB) can handle 2x the RTX 3050 batch

# ─── Helpers ───────────────────────────────────────────────────────────────────

def build_command(temperature: str, seed: int, batch_size: int, output_dir: str) -> list:
    """Build the command to run a single experiment."""
    cmd = [
        sys.executable, EXPERIMENT_RUNNER,
        "--algorithm", BASE_CONFIG["algorithm"],
        "--model", BASE_CONFIG["model"],
        "--reward", BASE_CONFIG["reward"],
        "--penalty", BASE_CONFIG["penalty"],
        "--temperature", temperature,
        "--problem", PROBLEM,
        "--max_steps", str(MAX_STEPS),
        "--batch_size", str(batch_size),
        "--patience", "999",
        "--seeds", str(seed),
        "--output_dir", output_dir,
        "--no_wandb",
    ]
    return cmd


def run_experiment(temperature: str, seed: int, batch_size: int, output_dir: str,
                  dry_run: bool = False) -> dict:
    """Run a single experiment and return results."""
    cmd = build_command(temperature, seed, batch_size, output_dir)
    label = f"[Test2] {PROBLEM} | temp={temperature} | seed={seed}"

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"temperature": temperature, "seed": seed, "best_r2": 0.0, "status": "dry_run"}

    result = subprocess.run(cmd, capture_output=False)

    # Find result file
    search_dir = Path(output_dir)
    jsons = sorted(search_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if jsons:
        latest = jsons[-1]
        with open(latest) as f:
            data = json.load(f)
        best_r2 = data.get("best_r2", -999.0)
        best_expr = data.get("best_expression", "N/A")
        total_steps = data.get("total_steps", 0)
        print(f"  → best_r2={best_r2:.4f} | steps={total_steps} | expr={best_expr}")
        return {"temperature": temperature, "seed": seed, "best_r2": best_r2,
                "best_expression": best_expr, "total_steps": total_steps, "file": str(latest)}

    print("  → WARNING: No result file found")
    return {"temperature": temperature, "seed": seed, "best_r2": -999.0, "status": "no_result"}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test 2: Nguyen-5 Debug")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE} for T4; use 256 for RTX 3050)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output dir (use Drive path on Colab: /content/drive/MyDrive/...)")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  PRE-PHASE B — TEST 2: NGUYEN-5 FAILURE DEBUG")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Problem: {PROBLEM}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  Total runs: {len(SEEDS) * len(TEMPERATURES)}")
    print(f"{'='*60}")

    all_results = []
    for temp in TEMPERATURES:
        for seed in SEEDS:
            result = run_experiment(temp, seed, args.batch_size, args.output_dir,
                                   dry_run=args.dry_run)
            all_results.append(result)

    # Save summary
    if not args.dry_run:
        summary_path = Path(args.output_dir) / "test2_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "test": "nguyen5_debug",
                "config": BASE_CONFIG,
                "problem": PROBLEM,
                "seeds": SEEDS,
                "temperatures": TEMPERATURES,
                "max_steps": MAX_STEPS,
                "results": all_results,
                "timestamp": start.isoformat(),
            }, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"  TEST 2 DONE in {elapsed:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
