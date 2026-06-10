#!/usr/bin/env python3
"""
Test 3: Convergence Profiling at 200 Steps

Goal: Profile the learning curve shape of the winner config to determine:
- Whether R² is still improving at step 200
- Steps-to-plateau for each problem difficulty
- Smart patience value for Phase B

Runs: 1 seed × 3 problems = 3 experiments (200 steps each, NO early stopping)
Key metric: R² trajectory shape — still rising? plateaued? oscillating?
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
OUTPUT_DIR = "results/pre_phase_b/test3_convergence"

WINNER_CONFIG = {
    "algorithm": "bon_ppo",
    "model": "augustocsc/gpt2_base_infix_682k",
    "reward": "r2_clipped",
    "penalty": "gradient",
    "temperature": "cosine_annealing",
}

SEED = 42
PROBLEMS = ["nguyen_1", "nguyen_5", "nguyen_9"]
MAX_STEPS = 200
BATCH_SIZE = 512  # T4 (16GB) — larger batch = more candidates per RL step

# ─── Helpers ───────────────────────────────────────────────────────────────────

def build_command(problem: str, batch_size: int) -> list:
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
        "--batch_size", str(batch_size),
        "--patience", "999",  # Disable early stopping — we want the full curve
        "--seeds", str(SEED),
        "--output_dir", OUTPUT_DIR,
        "--no_wandb",
    ]
    return cmd


def run_experiment(problem: str, batch_size: int, dry_run: bool = False) -> dict:
    """Run a single experiment and return results."""
    cmd = build_command(problem, batch_size)
    label = f"[Test3] {problem} | seed={SEED} | steps={MAX_STEPS}"

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Config: {WINNER_CONFIG['algorithm']} | {WINNER_CONFIG['reward']} | {WINNER_CONFIG['temperature']}")
    print(f"  NOTE: Early stopping DISABLED (patience=999)")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"problem": problem, "best_r2": 0.0, "status": "dry_run"}

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
        total_steps = data.get("total_steps", 0)

        # Extract learning curve from history
        history = data.get("history", [])
        r2_trajectory = [(h["step"], h["best_r2"]) for h in history]
        reward_trajectory = [(h["step"], h["mean_reward"]) for h in history]

        print(f"  → best_r2={best_r2:.4f} | steps_ran={total_steps} | expr={best_expr}")
        if r2_trajectory:
            first_r2 = r2_trajectory[0][1]
            last_r2 = r2_trajectory[-1][1]
            print(f"  → R² trajectory: {first_r2:.4f} (step 0) → {last_r2:.4f} (step {r2_trajectory[-1][0]})")

        return {
            "problem": problem, "best_r2": best_r2,
            "best_expression": best_expr, "total_steps": total_steps,
            "r2_trajectory": r2_trajectory,
            "reward_trajectory": reward_trajectory,
            "file": str(latest),
        }

    print("  → WARNING: No result file found")
    return {"problem": problem, "best_r2": -999.0, "status": "no_result"}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test 3: Convergence Profile (200 steps)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE} for T4; use 256 for RTX 3050)")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  PRE-PHASE B — TEST 3: CONVERGENCE PROFILING (200 STEPS)")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Seed: {SEED}")
    print(f"  Problems: {PROBLEMS}")
    print(f"  Max steps: {MAX_STEPS} (early stopping DISABLED)")
    print(f"  Total runs: {len(PROBLEMS)}")
    print(f"{'='*60}")

    all_results = []
    for problem in PROBLEMS:
        result = run_experiment(problem, args.batch_size, dry_run=args.dry_run)
        all_results.append(result)

    # Save summary
    if not args.dry_run:
        summary_path = Path(OUTPUT_DIR) / "test3_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "test": "convergence_200",
                "config": WINNER_CONFIG,
                "seed": SEED,
                "problems": PROBLEMS,
                "max_steps": MAX_STEPS,
                "results": all_results,
                "timestamp": start.isoformat(),
            }, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"  TEST 3 DONE in {elapsed:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
