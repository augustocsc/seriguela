#!/usr/bin/env python3
"""Re-score Phase 4 (single-seed) results with the fixed Expression class.

Companion to rescore_phase2.py, for the phase_4_local/phase_4 storage layout
(single seed=42 per problem, not 5-seed like Phase 2). Corrects the SOURCE
JSON files (results_latest.json + aggregate_*.json) in place -- editing only
the derived mother_table.csv (as an earlier pass did) does not stick, since
build_mother_table.py / srbench_comparison.py re-derive it from these JSONs
every time they run.

Recovery pool = the run's own best_expression + its own top_expressions
(same-run only, single seed -- no cross-seed borrowing needed here since
there is only one seed per problem in Phase 4).

Usage:
    python 4_analysis/rescore_phase4.py            # dry run
    python 4_analysis/rescore_phase4.py --apply     # write corrected files
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2_training" / "reinforcement"))
sys.path.insert(0, str(REPO / "classes"))
from utils.feynman_loader import load_benchmark_data, is_feynman_or_strogatz  # noqa: E402
from expression import Expression  # noqa: E402

RESULT_DIRS = ["results/phase_4_local", "results/phase_4"]
SOLVE_THRESHOLD = 0.999
EXCLUDE = {"nguyen_3", "nguyen_5", "nguyen_7", "nguyen_9"}  # handled by rescore_phase2.py

NGUYEN_BENCHMARKS = {
    "nguyen_1": {"equation": "x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_2": {"equation": "x**4 + x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_4": {"equation": "x**6 + x**5 + x**4 + x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_6": {"equation": "sin(x) + sin(x + x**2)", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_8": {"equation": "sqrt(x)", "vars": ["x_1"], "domain": (0, 4), "n_samples": 100},
    "nguyen_10": {"equation": "2*sin(x)*cos(y)", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
    "nguyen_11": {"equation": "x**y", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
    "nguyen_12": {"equation": "x**4 - x**3 + y**2/2 - y", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
}


def generate_nguyen_data(problem, seed):
    b = NGUYEN_BENCHMARKS[problem]
    n_vars = len(b["vars"])
    np.random.seed(seed)
    x = np.random.uniform(b["domain"][0], b["domain"][1], (b["n_samples"], n_vars))
    local_vars = {}
    for i, var_name in enumerate(b["vars"]):
        local_vars[var_name.replace("_", "")] = x[:, i]
    local_vars["x"] = x[:, 0]
    local_vars["y"] = x[:, 1] if n_vars >= 2 else None
    local_vars.update({"sin": np.sin, "cos": np.cos, "tan": np.tan,
                        "sqrt": np.sqrt, "log": np.log, "exp": np.exp})
    y = eval(b["equation"], {"__builtins__": None}, local_vars)
    return x, y


def get_train_test(problem, seed):
    if is_feynman_or_strogatz(problem):
        d = load_benchmark_data(problem, seed=seed, test_fraction=0.25)
        return d["train"]["x"], d["train"]["y"], d["test"]["x"], d["test"]["y"]
    x_tr, y_tr = generate_nguyen_data(problem, seed)
    x_te, y_te = generate_nguyen_data(problem, seed + 10000)
    return x_tr, y_tr, x_te, y_te


def score(expr_str, x, y):
    try:
        e = Expression(expr_str, is_prefix=False)
        y_pred = e.evaluate(x)
        if y_pred is None or not np.all(np.isfinite(y_pred)):
            return None
        return float(r2_score(y, y_pred))
    except Exception:
        return None


def recover(problem, seed, result_obj):
    pool = [result_obj.get("best_expression", "")]
    for te in result_obj.get("top_expressions", []) or []:
        e = te.get("expression")
        if e:
            pool.append(e)
    pool = list(dict.fromkeys(pool))

    try:
        x_tr, y_tr, x_te, y_te = get_train_test(problem, seed)
    except Exception as e:
        print(f"  [SKIP] {problem}: data regen failed ({e})")
        return None

    best = None
    for expr in pool:
        r2_tr = score(expr, x_tr, y_tr)
        if r2_tr is None:
            continue
        if best is None or r2_tr > best[0]:
            r2_te = score(expr, x_te, y_te)
            best = (r2_tr, expr, r2_te)
    if best is None:
        return None
    r2_tr, expr, r2_te = best
    return expr, r2_tr, r2_te


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak_prebugfix")
    if not bak.exists():
        shutil.copy2(path, bak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    log = []
    for base in RESULT_DIRS:
        for results_latest in glob.glob(str(REPO / base / "*/bon_ppo/**/results_latest.json"), recursive=True):
            fpath = Path(results_latest)
            parts = fpath.parts
            i = parts.index(base.split("/")[-1])
            problem = parts[i + 1]
            if problem in EXCLUDE:
                continue
            d = json.load(open(fpath, encoding="utf-8"))
            expr = d.get("best_expression", "")
            if "**C" not in expr:
                continue
            seed = d.get("config", {}).get("seed", 42)
            # seed isn't always in config; fall back to path
            seed_dirs = [p for p in parts if p.startswith("seed_")]
            if seed_dirs:
                seed = int(seed_dirs[0].replace("seed_", ""))

            rec = recover(problem, seed, d)
            if rec is None:
                log.append((problem, fpath, expr, None, None, None, "SKIP"))
                continue
            new_expr, new_best, new_test = rec
            log.append((problem, fpath, expr, new_expr, (d.get("best_r2"), new_best),
                        (d.get("test_r2"), new_test), ""))
            if args.apply:
                backup(fpath)
                d["best_expression"] = new_expr
                d["best_r2"] = new_best
                if d.get("test_r2") is not None:
                    d["test_r2"] = new_test
                json.dump(d, open(fpath, "w", encoding="utf-8"), indent=2)

                # sync the matching aggregate file
                agg_glob = glob.glob(str(fpath.parents[3] / ".." / "aggregate_*.json"))
                agg_glob = glob.glob(str(Path(base) / problem / "bon_ppo" / "aggregate_*.json"))
                for agg_path in agg_glob:
                    ad = json.load(open(agg_path, encoding="utf-8"))
                    touched = False
                    for r in ad.get("individual_results", []):
                        if r.get("best_expression") == expr:
                            r["best_expression"] = new_expr
                            r["best_r2"] = new_best
                            if r.get("test_r2") is not None:
                                r["test_r2"] = new_test
                            touched = True
                    if touched:
                        ind = ad["individual_results"]
                        best_r2s = [x.get("best_r2") for x in ind if x.get("best_r2") is not None]
                        test_r2s = [x.get("test_r2") for x in ind if x.get("test_r2") is not None]
                        if best_r2s:
                            ad["mean_best_r2"] = float(np.mean(best_r2s))
                            ad["std_best_r2"] = float(np.std(best_r2s))
                        if test_r2s:
                            ad["mean_test_r2"] = float(np.mean(test_r2s))
                            ad["solution_rate"] = float(sum(1 for v in test_r2s if v >= SOLVE_THRESHOLD) / len(test_r2s))
                            ad["n_solved"] = int(sum(1 for v in test_r2s if v >= SOLVE_THRESHOLD))
                        backup(Path(agg_path))
                        json.dump(ad, open(agg_path, "w", encoding="utf-8"), indent=2)

    print(f"\n{'='*110}")
    print(f"{'problem':<22}{'old_best':>9} -> {'new_best':>9}   {'old_test':>9} -> {'new_test':>9}  new_expr")
    print("=" * 110)
    n_flip = 0
    for problem, fpath, old_expr, new_expr, best_pair, test_pair, note in log:
        if note:
            print(f"{problem:<22}{note}: {old_expr}")
            continue
        ob, nb = best_pair
        ot, nt = test_pair
        ob_s = f"{ob:.4f}" if ob is not None else "  --  "
        nb_s = f"{nb:.4f}" if nb is not None else "  --  "
        ot_s = f"{ot:.4f}" if ot is not None else "  --  "
        nt_s = f"{nt:.4f}" if nt is not None else "  --  "
        flip = (ob is not None and ob >= SOLVE_THRESHOLD) != (nb is not None and nb >= SOLVE_THRESHOLD)
        if flip:
            n_flip += 1
        flag = "  <-- SOLVED FLAG FLIPPED" if flip else ""
        print(f"{problem:<22}{ob_s:>9} -> {nb_s:>9}   {ot_s:>9} -> {nt_s:>9}  {new_expr[:45] if new_expr else '':<45}{flag}")

    n_scored = sum(1 for x in log if not x[-1])
    print(f"\n{len(log)} results touched ({n_scored} rescored); {n_flip} 'solved' flag flips.")
    if args.apply:
        print("[APPLIED] source JSON files + aggregates updated.")
    else:
        print("(dry run -- pass --apply to write)")


if __name__ == "__main__":
    main()
