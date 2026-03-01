#!/usr/bin/env python3
"""
Test 1: Multi-Seed Robustness Validation

Goal: Check if the winner config (bon_ppo + r2_clipped + cosine_annealing + gradient)
produces consistent results across multiple seeds.

Runs: 3 seeds × 3 problems = 9 experiments (50 steps each)
Key metric: σ(R²) across seeds. If > 0.1, the winner may be seed-dependent.
Estimated time: ~3 hours on notebook (WSL with RTX 3050)
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_RUNNER = "2_training/reinforcement/run_experiment.py"
OUTPUT_DIR = "results/pre_phase_b/test1_multi_seed"

WINNER_CONFIG = {
    "algorithm": "bon_ppo",
    "model": "augustocsc/gpt2_base_infix_682k",
    "reward": "r2_clipped",
    "penalty": "gradient",
    "temperature": "cosine_annealing",
}

SEEDS = [42, 123, 456]
PROBLEMS = ["nguyen_1", "nguyen_5", "nguyen_9"]
MAX_STEPS = 50
BATCH_SIZE = 256

# ─── Helpers ───────────────────────────────────────────────────────────────────

def build_command(problem: str, seed: int) -> list:
    """Build the command to run a single experiment."""
    cmd = [
        sys.executable, EXPERIMENT_RUNNER,
        "--algorithm", WINNER_CONFIG["algorithm"],
        "--model", WINNER_CONFIG["model"],
        "--reward", WINNER_CONFIG["reward"],
        "--penalty", WINNER_CONFIG["penalty"],
        "--temperature", WINNER_CONFIG["temperature"],
        "--problem", problem,
        "--max_steps", str(MAX_STEPS),
        "--batch_size", str(BATCH_SIZE),
        "--patience", "999",  # Don't early stop — we want full 50 steps
        "--seeds", str(seed),
        "--output_dir", OUTPUT_DIR,
        "--load_in_4bit",
        "--no_wandb",
    ]
    return cmd


def run_experiment(problem: str, seed: int, dry_run: bool = False) -> dict:
    """Run a single experiment and return results."""
    cmd = build_command(problem, seed)
    label = f"[Test1] {problem} | seed={seed}"

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Config: {WINNER_CONFIG['algorithm']} | {WINNER_CONFIG['reward']} | {WINNER_CONFIG['temperature']}")
    print(f"  Steps: {MAX_STEPS}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"problem": problem, "seed": seed, "best_r2": 0.0, "status": "dry_run"}

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
        return {"problem": problem, "seed": seed, "best_r2": best_r2,
                "best_expression": best_expr, "file": str(latest)}

    print("  → WARNING: No result file found")
    return {"problem": problem, "seed": seed, "best_r2": -999.0, "status": "no_result"}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test 1: Multi-Seed Robustness")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  PRE-PHASE B — TEST 1: MULTI-SEED ROBUSTNESS")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Problems: {PROBLEMS}")
    print(f"  Total runs: {len(SEEDS) * len(PROBLEMS)}")
    print(f"{'='*60}")

    all_results = []
    for problem in PROBLEMS:
        for seed in SEEDS:
            result = run_experiment(problem, seed, dry_run=args.dry_run)
            all_results.append(result)

    # Save summary
    if not args.dry_run:
        summary_path = Path(OUTPUT_DIR) / "test1_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "test": "multi_seed_robustness",
                "config": WINNER_CONFIG,
                "seeds": SEEDS,
                "problems": PROBLEMS,
                "max_steps": MAX_STEPS,
                "results": all_results,
                "timestamp": start.isoformat(),
            }, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"  TEST 1 DONE in {elapsed:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
