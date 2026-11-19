"""Pilot timing harness — measures real wall-clock cost and CONCURRENCY gains on a cloud GPU.

Phases (all runs use C=1 evaluation, the project default — uniform with phases 1b/1c):
  S  (solo):        pure_ppo nguyen_5, 500 steps          -> baseline s/step + PPO plateau scout
  C2 (2 concurrent): pure_grpo + bon_ppo nguyen_5, 500     -> GRPO/BoN scouts + 2x contention factor
  C3 (3 concurrent): 3x pure_ppo nguyen_5, 150 steps       -> 3x contention probe (marginal slowdown)
  P  (probes):       best_of_n throughput + feynman RL run -> baseline & Phase 4.2 per-run estimates

The 500-step runs double as the Fase 1.5 plateau scout, so this compute is not throwaway.

Usage (on the GPU instance, venv active, repo root):
    python experiments/run_pilot_timing.py                 # full pilot
    python experiments/run_pilot_timing.py --quick         # cap runs at 60 steps (smoke-level timing)
    python experiments/run_pilot_timing.py --skip-c3       # skip the 3x concurrency probe
    python experiments/run_pilot_timing.py --price 0.44    # GPU USD/h for cost projection

Outputs: results/pilot_timing/timing_report.json + timing_report.md
The report is rewritten after EVERY phase, so early termination still yields data.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "2_training" / "reinforcement" / "run_experiment.py"
OUT_DIR = REPO_ROOT / "results" / "pilot_timing"

COMMON = [
    "--model", "augustocsc/gpt2_base_infix_682k",
    "--reward", "sr_ic",
    "--penalty", "gradient",
    "--temperature", "cosine_annealing",
    "--prompt_type", "standard",
    "--batch_size", "1024",
    # patience alto = early stopping efetivamente OFF. O default (5) matou o
    # piloto v1 em 6-9 steps reais: RL parte de R²=0.0 e demora dezenas de
    # steps para melhorar — e o scout de plateau PRECISA da curva inteira.
    "--patience", "100000",
    "--no_wandb",
    "--no_resume",
]
QUICK_STEPS = 60


def make_cmd(name: str, algorithm: str, problem: str, steps: int, seed: int, extra: list) -> tuple:
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT),
        "--algorithm", algorithm,
        "--problem", problem,
        "--max_steps", str(steps),
        "--seeds", str(seed),
        "--output_dir", str(out),
        *COMMON, *extra,
    ]
    return cmd, out


def finalize(rec: dict, out: Path):
    """Enrich a finished run record with metrics parsed from its outputs.

    steps_done vem do aggregate (individual_results[].total_steps) — fonte
    exata. O regex em stdout é só fallback, com lookbehind para não casar
    "max_steps: 500" do dump de config (bug do piloto v1: s/step ficou 50x
    menor porque o denominador era o max_steps pedido, não o executado).
    """
    agg = out / f"aggregate_{rec['algorithm']}_{rec['problem']}_seed{rec['seed']}.json"
    if agg.exists():
        try:
            d = json.load(open(agg, encoding="utf-8"))
            rec["mean_best_r2"] = d.get("mean_best_r2")
            rec["mean_test_r2"] = d.get("mean_test_r2")
            ind = d.get("individual_results") or []
            steps = [r.get("total_steps") for r in ind if r.get("total_steps")]
            if steps:
                rec["steps_executed"] = max(steps)
        except Exception as e:  # noqa: BLE001
            rec["aggregate_parse_error"] = str(e)
    if "steps_executed" not in rec:
        try:
            txt = (out / "stdout.log").read_text(encoding="utf-8", errors="ignore")
            step_nums = [int(m) for m in re.findall(r"(?<![_\w])[Ss]tep[\s:=/]+(\d+)", txt)]
            if step_nums:
                rec["steps_executed"] = max(step_nums)
        except Exception:  # noqa: BLE001
            pass
    steps_done = rec.get("steps_executed") or rec["max_steps_requested"]
    rec["sec_per_step"] = round(rec["wall_seconds"] / max(steps_done, 1), 2)


def run_phase(phase: str, specs: list, quick: bool) -> list:
    """Run all specs of a phase CONCURRENTLY; return one record per run."""
    procs = []
    for (name, algorithm, problem, steps, seed, extra) in specs:
        steps = min(steps, QUICK_STEPS) if quick else steps
        cmd, out = make_cmd(name, algorithm, problem, steps, seed, extra)
        logf = open(out / "stdout.log", "w", encoding="utf-8")
        print(f"[{phase}] start {name} (steps={steps}): {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
        procs.append({
            "proc": p, "logf": logf, "t0": time.monotonic(), "out": out,
            "rec": {"phase": phase, "name": name, "algorithm": algorithm, "problem": problem,
                    "seed": seed, "max_steps_requested": steps, "concurrency": len(specs)},
        })
        time.sleep(20)  # stagger model loads to avoid a simultaneous download/VRAM spike

    records = []
    pending = list(procs)
    while pending:
        time.sleep(10)
        for item in pending[:]:
            if item["proc"].poll() is not None:
                wall = time.monotonic() - item["t0"]
                item["logf"].close()
                rec = item["rec"]
                rec["wall_seconds"] = round(wall, 1)
                rec["exit_code"] = item["proc"].returncode
                finalize(rec, item["out"])
                print(f"[{phase}] done {rec['name']}: {wall/60:.1f} min, ~{rec['sec_per_step']}s/step "
                      f"(exit={rec['exit_code']})", flush=True)
                records.append(rec)
                pending.remove(item)
    return records


def projections(records: list, price: float) -> dict:
    ok = [r for r in records if r.get("exit_code") == 0]
    by_conc = {}
    for r in ok:
        if r["algorithm"] in ("pure_ppo", "pure_grpo", "bon_ppo"):
            by_conc.setdefault(r["concurrency"], []).append(r["sec_per_step"])
    eff = {}
    base = None
    if by_conc.get(1):
        base = sum(by_conc[1]) / len(by_conc[1])
    for c, vals in sorted(by_conc.items()):
        mean_sps = sum(vals) / len(vals)
        runs_per_h = c * 3600 / (mean_sps * 200)  # 200-step seed-runs per hour at this concurrency
        eff[f"concurrency_{c}"] = {
            "mean_sec_per_step": round(mean_sps, 2),
            "slowdown_vs_solo": round(mean_sps / base, 2) if base else None,
            "seedruns_200steps_per_hour": round(runs_per_h, 2),
            "usd_per_seedrun": round(price / runs_per_h, 3) if runs_per_h else None,
        }
    proj = {"efficiency": eff, "usd_per_hour": price,
            "assumptions": {"phase2": "100 seed-runs x 200 steps", "phase42_optB": "150 seed-runs x 150 steps",
                            "overhead_factor": 1.15}}
    # Use the best measured concurrency for projections
    best = None
    for c, e in eff.items():
        if e["usd_per_seedrun"] is not None and (best is None or e["usd_per_seedrun"] < best[1]["usd_per_seedrun"]):
            best = (c, e)
    if best:
        c, e = best
        proj["best_config"] = c
        p2_h = 100 / e["seedruns_200steps_per_hour"] * 1.15
        proj["phase2_hours"] = round(p2_h, 1)
        proj["phase2_usd"] = round(p2_h * price, 2)
        fey = [r for r in ok if r["problem"].startswith("feynman")]
        sps_f = fey[0]["sec_per_step"] if fey else e["mean_sec_per_step"]
        p42_s = sps_f * 150 * 150 * 1.15 / (int(c.split("_")[1]))
        proj["phase42_optB_hours"] = round(p42_s / 3600, 1)
        proj["phase42_optB_usd"] = round(p42_s / 3600 * price, 2)
    return proj


def write_report(records: list, price: float, started: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proj = projections(records, price)
    report = {"started_utc": started, "updated_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_price_usd_h": price, "runs": records, "projections": proj}
    (OUT_DIR / "timing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Pilot Timing Report (C=1 evaluation, batch 1024)", "",
             f"Started: {started}  ", f"Updated: {report['updated_utc']}", "",
             "| phase | run | algo | problem | conc | steps | wall (min) | s/step | exit | best R2 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r['phase']} | {r['name']} | {r['algorithm']} | {r['problem']} | {r['concurrency']} "
            f"| {r.get('steps_executed', r['max_steps_requested'])} | {r['wall_seconds']/60:.1f} "
            f"| {r['sec_per_step']} | {r['exit_code']} | {r.get('mean_best_r2', '—')} |")
    lines += ["", "## Projections", "```json", json.dumps(proj, indent=2), "```", ""]
    (OUT_DIR / "timing_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report updated] {OUT_DIR / 'timing_report.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help=f"cap every run at {QUICK_STEPS} steps")
    ap.add_argument("--skip-c3", action="store_true", help="skip the 3x concurrency probe")
    ap.add_argument("--price", type=float, default=0.44, help="GPU USD/hour for cost projection")
    args = ap.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    records = []

    phases = [
        ("S", [("solo_pure_ppo_nguyen5", "pure_ppo", "nguyen_5", 500, 42, [])]),
        ("C2", [("c2_pure_grpo_nguyen5", "pure_grpo", "nguyen_5", 500, 42, []),
                ("c2_bon_ppo_nguyen5", "bon_ppo", "nguyen_5", 500, 42, ["--buffer_ratio", "0.2"])]),
    ]
    if not args.skip_c3:
        phases.append(("C3", [
            ("c3_pure_ppo_a", "pure_ppo", "nguyen_5", 150, 123, []),
            ("c3_pure_ppo_b", "pure_ppo", "nguyen_5", 150, 456, []),
            ("c3_pure_ppo_c", "pure_ppo", "nguyen_5", 150, 789, []),
        ]))
    phases.append(("P", [("probe_best_of_n", "best_of_n", "nguyen_5", 50, 42, [])]))
    phases.append(("P", [("probe_feynman_ppo", "pure_ppo", "feynman_I_12_2", 200, 42, [])]))

    for phase, specs in phases:
        records.extend(run_phase(phase, specs, args.quick))
        write_report(records, args.price, started)
    print("\nPILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
