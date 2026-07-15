#!/usr/bin/env python3
"""Re-score Phase 2 results (nguyen_3/5/7/9, 5 arms, 5 seeds) with the fixed
Expression class (see classes/expression.py -- the x**C -> x**2 hardcode).

Seed-scoped: each seed's corrected result is recovered ONLY from that same
seed's own discovered candidates (best_expression + top_expressions), never
borrowing from another seed. Crossing seeds would contaminate the seed-level
independence the Phase 2/3 statistics (ANOVA, Tukey, Cohen's d) rely on.

Two storage layouts in Phase 2:
  - pure_ppo/pure_grpo/bon_ppo/bon_grpo: per-seed results_latest.json files
    (path encodes the seed unambiguously: .../seed_XXXX/results_latest.json)
  - best_of_n: no per-seed file; only the aggregate's individual_results list
    (seed taken from the aggregate's own "seeds" field, position-matched)

After correcting the primary source for each arm, the aggregate_*.json
summary stats (mean_best_r2, std_best_r2, mean_test_r2, solution_rate,
n_solved, and each individual_results entry) are recomputed to stay
consistent, since build_mother_table.py / srbench_comparison.py read the
aggregate directly for nguyen_3/5/7/9.

Backs up every file it touches with a .bak_prebugfix suffix before writing.

Usage:
    python 4_analysis/rescore_phase2.py            # dry run, prints diff
    python 4_analysis/rescore_phase2.py --apply     # writes corrected files
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
sys.path.insert(0, str(REPO / "classes"))
from expression import Expression  # noqa: E402

P2 = REPO / "results" / "phase_2" / "base_infix"
SOLVE_THRESHOLD = 0.999

NGUYEN_BENCHMARKS = {
    "nguyen_3": {"equation": "x**5 + x**4 + x**3 + x**2 + x", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_5": {"equation": "sin(x**2) * cos(x) - 1", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_7": {"equation": "log(x + 1) + log(x**2 + 1)", "vars": ["x_1"], "domain": (0, 2), "n_samples": 100},
    "nguyen_9": {"equation": "sin(x) + sin(y**2)", "vars": ["x_1", "x_2"], "domain": (0, 2), "n_samples": 100},
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


_data_cache = {}


def train_test(problem, seed):
    key = (problem, seed)
    if key not in _data_cache:
        x_tr, y_tr = generate_nguyen_data(problem, seed)
        x_te, y_te = generate_nguyen_data(problem, seed + 10000)
        _data_cache[key] = (x_tr, y_tr, x_te, y_te)
    return _data_cache[key]


def score(expr_str, x, y):
    try:
        e = Expression(expr_str, is_prefix=False)
        y_pred = e.evaluate(x)
        if y_pred is None or not np.all(np.isfinite(y_pred)):
            return None
        return float(r2_score(y, y_pred))
    except Exception:
        return None


def recover_one(problem, seed, result_obj):
    """Seed-scoped recovery: pool = this result's own best_expression + its
    own top_expressions. Returns (new_expr, new_best_r2, new_test_r2) or None
    if nothing in the pool beats -inf (should not happen in practice)."""
    old_expr = result_obj.get("best_expression", "")
    pool = [old_expr]
    for te in result_obj.get("top_expressions", []) or []:
        e = te.get("expression")
        if e:
            pool.append(e)
    pool = list(dict.fromkeys(pool))  # dedupe, keep order

    x_tr, y_tr, x_te, y_te = train_test(problem, seed)
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


def process_per_seed_files(apply, log):
    """pure_ppo, pure_grpo, bon_ppo, bon_grpo -- per-seed results_latest.json."""
    for algo_dir in glob.glob(str(P2 / "*" / "*")):
        p = Path(algo_dir)
        problem, algo = p.parts[-2], p.parts[-1]
        if algo == "best_of_n" or problem not in NGUYEN_BENCHMARKS:
            continue
        for f in glob.glob(str(p / "**" / "seed_*" / "results_latest.json"), recursive=True):
            fpath = Path(f)
            seed = int(fpath.parent.name.replace("seed_", ""))
            d = json.load(open(fpath, encoding="utf-8"))
            expr = d.get("best_expression", "")
            if "**C" not in expr:
                continue
            rec = recover_one(problem, seed, d)
            if rec is None:
                log.append((problem, algo, seed, expr, None, None, None, "SKIP (no valid candidate)"))
                continue
            new_expr, new_best, new_test = rec
            log.append((problem, algo, seed, expr, new_expr,
                        (d.get("best_r2"), new_best), (d.get("test_r2"), new_test), ""))
            if apply:
                backup(fpath)
                d["best_expression"] = new_expr
                d["best_r2"] = new_best
                if d.get("test_r2") is not None:
                    d["test_r2"] = new_test
                json.dump(d, open(fpath, "w", encoding="utf-8"), indent=2)


def process_aggregates(apply, log):
    """best_of_n individual_results (only record) + resync every aggregate's
    individual_results/summary stats for all 5 arms (so aggregate files stay
    consistent with the corrected per-seed sources)."""
    for agg_path in glob.glob(str(P2 / "*" / "*" / "aggregate_*.json")):
        fpath = Path(agg_path)
        p = fpath.parent
        problem, algo = p.parts[-2], p.parts[-1]
        if problem not in NGUYEN_BENCHMARKS:
            continue
        d = json.load(open(fpath, encoding="utf-8"))
        seeds_list = d.get("seeds") or []
        touched = False

        for k, r in enumerate(d.get("individual_results", [])):
            seed = r.get("seed")
            if seed is None:
                seed = seeds_list[k] if k < len(seeds_list) else None
            if seed is None:
                continue

            if algo == "best_of_n":
                # aggregate IS the primary source for best_of_n
                expr = r.get("best_expression", "")
                if "**C" not in expr:
                    continue
                rec = recover_one(problem, seed, r)
                if rec is None:
                    log.append((problem, algo, seed, expr, None, None, None, "SKIP"))
                    continue
                new_expr, new_best, new_test = rec
                log.append((problem, algo, seed, expr, new_expr,
                            (r.get("best_r2"), new_best), (r.get("test_r2"), new_test), ""))
                if apply:
                    r["best_expression"] = new_expr
                    r["best_r2"] = new_best
                    if r.get("test_r2") is not None:
                        r["test_r2"] = new_test
                    touched = True
            else:
                # resync from the (already corrected) per-seed results_latest.json
                seed_glob = glob.glob(str(REPO / f"results/phase_2/base_infix/{problem}/{algo}/**/seed_{seed}/results_latest.json"), recursive=True)
                if not seed_glob:
                    continue
                sd = json.load(open(seed_glob[0], encoding="utf-8"))
                if r.get("best_expression") != sd.get("best_expression"):
                    if apply:
                        r["best_expression"] = sd.get("best_expression")
                        r["best_r2"] = sd.get("best_r2")
                        if r.get("test_r2") is not None:
                            r["test_r2"] = sd.get("test_r2")
                        touched = True

        if touched and apply:
            # recompute aggregate summary stats from corrected individual_results
            ind = d["individual_results"]
            best_r2s = [x.get("best_r2") for x in ind if x.get("best_r2") is not None]
            test_r2s = [x.get("test_r2") for x in ind if x.get("test_r2") is not None]
            train_r2s = [x.get("mean_r2") for x in ind if x.get("mean_r2") is not None]
            if best_r2s:
                d["mean_best_r2"] = float(np.mean(best_r2s))
                d["std_best_r2"] = float(np.std(best_r2s))
            if test_r2s:
                d["mean_test_r2"] = float(np.mean(test_r2s))
                d["std_test_r2"] = float(np.std(test_r2s))
                d["max_test_r2"] = float(np.max(test_r2s))
                d["min_test_r2"] = float(np.min(test_r2s))
                d["median_test_r2"] = float(np.median(test_r2s))
                d["solution_rate"] = float(sum(1 for v in test_r2s if v >= SOLVE_THRESHOLD) / len(test_r2s))
                d["n_solved"] = int(sum(1 for v in test_r2s if v >= SOLVE_THRESHOLD))
            backup(fpath)
            json.dump(d, open(fpath, "w", encoding="utf-8"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    log = []
    process_per_seed_files(args.apply, log)
    process_aggregates(args.apply, log)

    print(f"\n{'='*120}")
    print(f"{'problem':<12}{'algo':<12}{'seed':>6}  {'old_best':>9} -> {'new_best':>9}   "
          f"{'old_test':>9} -> {'new_test':>9}  new_expr")
    print("=" * 120)
    n_flip = 0
    for prob, algo, seed, old_expr, new_expr, best_pair, test_pair, note in log:
        if note:
            print(f"{prob:<12}{algo:<12}{seed:>6}  {note}: {old_expr}")
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
        print(f"{prob:<12}{algo:<12}{seed:>6}  {ob_s:>9} -> {nb_s:>9}   {ot_s:>9} -> {nt_s:>9}  "
              f"{new_expr[:45] if new_expr else '':<45}{flag}")

    n_scored = sum(1 for x in log if not x[-1])
    print(f"\n{len(log)} individual results touched ({n_scored} rescored, "
          f"{len(log)-n_scored} skipped); {n_flip} 'solved' flag flips.")
    if args.apply:
        print("\n[APPLIED] per-seed results_latest.json + aggregate files updated (.bak_prebugfix backups kept).")
        print("Next: re-run analyze_phase2.py to regenerate master_table/ANOVA.")
    else:
        print("\n(dry run -- pass --apply to write corrected files)")


if __name__ == "__main__":
    main()
