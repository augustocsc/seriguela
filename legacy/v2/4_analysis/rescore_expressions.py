#!/usr/bin/env python3
"""Re-score every discovered expression in the mother table with the FIXED
Expression class (classes/expression.py no longer hardcodes x**C -> x**2).

Does not re-run RL. For each row whose reported "found" expression contains
"**C" (i.e. could only have looked correct because of the bug), this script:

  1. Locates the original run(s) for that problem (all seeds, if multi-seed).
  2. Builds a recovery pool = {reported best_expression} U {each run's
     top_expressions} -- the same candidates the run itself surfaced, so nothing
     outside what was actually discovered is invented.
  3. Re-scores every candidate in the pool against freshly regenerated
     train/test data (same seed/domain run_experiment.py used) with the FIXED
     evaluator, and keeps the true best.

This matters because the buggy evaluator was also used *during* candidate
ranking/selection -- an expression that only looked good by exploiting the
exponent hack could have out-ranked a genuinely correct sibling candidate
from the very same run. Re-scoring only the single previously-reported
"best_expression" would unfairly report those runs as total failures.

Writes results/mother_table.csv in place (--apply) or prints a dry-run diff.

Usage:
    python 4_analysis/rescore_expressions.py            # dry run
    python 4_analysis/rescore_expressions.py --apply     # overwrite mother_table.csv
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "2_training" / "reinforcement"))
sys.path.insert(0, str(REPO / "classes"))

from utils.feynman_loader import load_benchmark_data, is_feynman_or_strogatz  # noqa: E402
from expression import Expression  # noqa: E402

MOTHER_CSV = REPO / "results" / "mother_table.csv"
RESULT_DIRS = ["results/phase_4_local", "results/phase_4", "results/phase_2"]
SOLVE_THRESHOLD = 0.999

NGUYEN_BENCHMARKS = {
    "nguyen_1": {"equation": "x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_2": {"equation": "x**4 + x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_3": {"equation": "x**5 + x**4 + x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_4": {"equation": "x**6 + x**5 + x**4 + x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_5": {"equation": "sin(x**2) * cos(x) - 1", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_6": {"equation": "sin(x) + sin(x + x**2)", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_7": {"equation": "log(x + 1) + log(x**2 + 1)", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_8": {"equation": "sqrt(x)", "vars": ["x_1"], "domain": (0, 4), "n_samples": 100},
    "nguyen_9": {"equation": "sin(x) + sin(y**2)", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
    "nguyen_10": {"equation": "2*sin(x)*cos(y)", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
    "nguyen_11": {"equation": "x**y", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
    "nguyen_12": {"equation": "x**4 - x**3 + y**2/2 - y", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
}


def generate_nguyen_data(problem, seed=None):
    benchmark = NGUYEN_BENCHMARKS[problem]
    n_vars = len(benchmark["vars"])
    n_samples = benchmark["n_samples"]
    domain = benchmark["domain"]
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(domain[0], domain[1], (n_samples, n_vars))
    local_vars = {}
    for i, var_name in enumerate(benchmark["vars"]):
        local_vars[var_name.replace("_", "")] = x[:, i]
    local_vars["x"] = x[:, 0]
    local_vars["y"] = x[:, 1] if n_vars >= 2 else None
    safe_funcs = {"sin": np.sin, "cos": np.cos, "tan": np.tan,
                  "sqrt": np.sqrt, "log": np.log, "exp": np.exp}
    local_vars.update(safe_funcs)
    y = eval(benchmark["equation"], {"__builtins__": None}, local_vars)
    return x, y


def get_train_test(problem, seed):
    if is_feynman_or_strogatz(problem):
        d = load_benchmark_data(problem, seed=seed, test_fraction=0.25)
        return d["train"]["x"], d["train"]["y"], d["test"]["x"], d["test"]["y"]
    x_tr, y_tr = generate_nguyen_data(problem, seed=seed)
    x_te, y_te = generate_nguyen_data(problem, seed=seed + 10000)
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


def load_individual_results(problem):
    """All individual_results entries (across every seed/aggregate found) for
    a problem's bon_ppo runs, each tagged with its seed."""
    out = []
    for base in RESULT_DIRS:
        for agg in glob.glob(str(REPO / base / f"**/{problem}/bon_ppo/aggregate_*.json"),
                              recursive=True):
            try:
                d = json.load(open(agg, encoding="utf-8"))
                for r in d.get("individual_results", []):
                    out.append(r)
            except Exception:
                pass
        if out:
            return out  # first RESULT_DIRS hit wins, same priority as build_mother_table.py
    return out


def recover_best(problem, reported_expr):
    """Search the recovery pool (reported best_expression + every run's
    top_expressions, across all seeds) and return the true best candidate
    under the fixed evaluator. Returns (expr, seed, train_r2, test_r2)."""
    individuals = load_individual_results(problem)
    if not individuals:
        return None

    candidates = []  # (expr_str, seed)
    for ind in individuals:
        seed = ind.get("seed", 42)
        be = ind.get("best_expression")
        if be:
            candidates.append((be, seed))
        for te in ind.get("top_expressions", []) or []:
            expr = te.get("expression")
            if expr:
                candidates.append((expr, seed))

    if not candidates:
        return None

    # group by seed to regenerate data once per seed
    by_seed = {}
    for expr, seed in candidates:
        by_seed.setdefault(seed, []).append(expr)

    best = None  # (train_r2, expr, seed, test_r2)
    for seed, exprs in by_seed.items():
        try:
            x_tr, y_tr, x_te, y_te = get_train_test(problem, seed)
        except Exception:
            continue
        seen = set()
        for expr in exprs:
            if expr in seen:
                continue
            seen.add(expr)
            r2_tr = score(expr, x_tr, y_tr)
            if r2_tr is None:
                continue
            if best is None or r2_tr > best[0]:
                r2_te = score(expr, x_te, y_te)
                best = (r2_tr, expr, seed, r2_te)

    if best is None:
        return None
    train_r2, expr, seed, test_r2 = best
    return expr, seed, train_r2, test_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(MOTHER_CSV, encoding="utf-8")))
    changed = []
    for row in rows:
        if row["status"] != "done" or not row.get("found"):
            continue
        if "**C" not in row["found"]:
            continue

        problem = row["problem"]
        old_best = float(row["best_r2"]) if row["best_r2"] else None
        old_test = float(row["test_r2"]) if row["test_r2"] else None
        old_expr = row["found"]

        result = recover_best(problem, old_expr)
        if result is None:
            print(f"[SKIP] {problem}: could not recover/regenerate data")
            continue
        new_expr, seed, new_best, new_test = result

        changed.append({
            "problem": problem, "old_expr": old_expr, "new_expr": new_expr, "seed": seed,
            "old_best": old_best, "new_best": new_best,
            "old_test": old_test, "new_test": new_test,
            "old_solved": row["solved"],
            "new_solved": "yes" if (new_best is not None and new_best >= SOLVE_THRESHOLD) else "no",
            "recovered": new_expr != old_expr,
        })

        row["found"] = new_expr[:60]
        row["best_r2"] = round(new_best, 4) if new_best is not None else None
        row["test_r2"] = round(new_test, 4) if new_test is not None else None
        row["solved"] = "yes" if (new_best is not None and new_best >= SOLVE_THRESHOLD) else "no"

    print(f"\n{'='*118}")
    print(f"{'problem':<20}{'old_best':>9} -> {'new_best':>9}   {'old':>4}->{'new':>4}  {'recovered?':<11} new_expr")
    print("=" * 118)
    for c in changed:
        ob = f"{c['old_best']:.4f}" if c['old_best'] is not None else "  --  "
        nb = f"{c['new_best']:.4f}" if c['new_best'] is not None else "  --  "
        rec = "YES (pool)" if c["recovered"] else "no"
        flag = "  <-- FLIPPED" if c["old_solved"] != c["new_solved"] else ""
        print(f"{c['problem']:<20}{ob:>9} -> {nb:>9}   {c['old_solved']:>4}->{c['new_solved']:>4}  "
              f"{rec:<11} {c['new_expr'][:55]}{flag}")

    n_flipped = sum(1 for c in changed if c["old_solved"] != c["new_solved"])
    n_recovered = sum(1 for c in changed if c["recovered"])
    n_still_solved = sum(1 for c in changed if c["new_solved"] == "yes")
    print(f"\n{len(changed)} rows re-scored; {n_recovered} recovered a better candidate from the pool; "
          f"{n_flipped} 'solved' flag changed; {n_still_solved} still solved after correction.")

    if args.apply:
        cols = list(rows[0].keys())
        with open(MOTHER_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[APPLIED] {MOTHER_CSV} overwritten with corrected values.")
    else:
        print("\n(dry run -- pass --apply to overwrite results/mother_table.csv)")


if __name__ == "__main__":
    main()
