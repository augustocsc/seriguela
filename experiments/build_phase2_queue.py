#!/usr/bin/env python3
"""Generate the Phase 2 experiment queue (THESIS_PLAN.md §4, Fase 2).

Design (controlled, no reward confound):
  - 5 algorithms : best_of_n, pure_ppo, pure_grpo, bon_ppo, bon_grpo
  - 4 problems   : nguyen_3, nguyen_5, nguyen_7, nguyen_9
  - 5 seeds      : 42 123 456 789 1011  (bundled per entry via --seeds nargs=+)
  - reward sr_ic / penalty gradient / temperature cosine_annealing / prompt standard
  - C=1 evaluation (project default — uniform with phases 1b/1c, see
    docs/reports/THESIS_PLAN.md and the constant-fitting decision)

MAX_STEPS defaults to 200 but MUST be revisited after the RunPod pilot
(results/pilot_timing/timing_report.md) — pass --max-steps with the value the
plateau scout supports.

Usage:
    python experiments/build_phase2_queue.py                  # preview (dry-run)
    python experiments/build_phase2_queue.py --write          # append to queue.yaml
    python experiments/build_phase2_queue.py --max-steps 300 --write
    python experiments/build_phase2_queue.py --model augustocsc/gpt2_base_infix_682k --write

Idempotent: entries whose id already exists in queue.yaml are skipped.
After writing, validate with:  python experiments/test_smoke.py
"""
import argparse
import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = REPO_ROOT / "experiments" / "queue.yaml"

ALGORITHMS = ["best_of_n", "pure_ppo", "pure_grpo", "bon_ppo", "bon_grpo"]
PROBLEMS = ["nguyen_3", "nguyen_5", "nguyen_7", "nguyen_9"]
SEEDS = [42, 123, 456, 789, 1011]


def short_model(model: str) -> str:
    """augustocsc/gpt2_base_infix_682k -> base_infix"""
    name = model.split("/")[-1]
    return name.replace("gpt2_", "").replace("_682k", "")


def build_entries(model: str, max_steps: int, bon_steps: int, batch_size: int) -> list:
    today = datetime.date.today().isoformat()
    entries = []
    for problem in PROBLEMS:
        for algo in ALGORITHMS:
            steps = bon_steps if algo == "best_of_n" else max_steps
            args = {
                "algorithm": algo,
                "model": model,
                "problem": problem,
                "seeds": SEEDS,
                "max_steps": steps,
                "batch_size": batch_size,
                "reward": "sr_ic",
                "penalty": "gradient",
                "temperature": "cosine_annealing",
                "prompt_type": "standard",
                "no_wandb": True,
            }
            if algo in ("bon_ppo", "bon_grpo"):
                args["buffer_ratio"] = 0.2
            mshort = short_model(model)
            entries.append({
                "id": f"phase2_{mshort}_{algo}_{problem}",
                "phase": 2,
                "status": "pending",
                "created": today,
                "args": args,
                "output_dir": f"results/phase_2/{mshort}/{problem}/{algo}/",
                # rough per-entry estimate; refresh after the pilot
                "estimated_minutes": 25 if algo == "best_of_n" else 5 * max_steps // 10,
            })
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="augustocsc/gpt2_base_infix_682k",
                    help="HF repo of the SFT model selected in Phase 1")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="RL steps per seed-run (set from the pilot plateau scout)")
    ap.add_argument("--bon-steps", type=int, default=None,
                    help="best_of_n batches per seed-run (default: same as --max-steps "
                         "for an equal sample budget at equal batch_size)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--write", action="store_true",
                    help="append to experiments/queue.yaml (default: dry-run preview)")
    args = ap.parse_args()

    bon_steps = args.bon_steps if args.bon_steps is not None else args.max_steps
    entries = build_entries(args.model, args.max_steps, bon_steps, args.batch_size)

    data = yaml.safe_load(QUEUE_FILE.read_text(encoding="utf-8"))
    existing_ids = {e["id"] for e in data.get("queue", [])}
    new = [e for e in entries if e["id"] not in existing_ids]
    dup = len(entries) - len(new)

    total_seedruns = len(new) * len(SEEDS)
    print(f"Phase 2 design: {len(ALGORITHMS)} algos x {len(PROBLEMS)} problems x "
          f"{len(SEEDS)} seeds = {len(ALGORITHMS)*len(PROBLEMS)*len(SEEDS)} seed-runs")
    print(f"New queue entries: {len(new)} ({dup} already present, skipped) "
          f"-> {total_seedruns} seed-runs to execute")
    for e in new:
        print(f"  + {e['id']} (max_steps={e['args']['max_steps']})")

    if not args.write:
        print("\nDry-run. Re-run with --write to append to experiments/queue.yaml")
        return

    data["queue"].extend(new)
    data.setdefault("meta", {})["updated"] = datetime.date.today().isoformat()
    QUEUE_FILE.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nWrote {len(new)} entries to {QUEUE_FILE}")
    print("Validate with: python experiments/test_smoke.py")


if __name__ == "__main__":
    main()
