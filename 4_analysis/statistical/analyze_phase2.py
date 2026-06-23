#!/usr/bin/env python3
"""Fase 3 — análise profunda dos resultados da Fase 2 (ver docs/reports/PHASE3_ANALYSIS_PLAN.md).

Roda local (CPU). Tolera dados parciais (seeds faltando) — útil para
acompanhamento durante a Fase 2; a análise estatística completa exige 5 seeds.

Uso:
    python 4_analysis/statistical/analyze_phase2.py                # tabela + dinâmica
    python 4_analysis/statistical/analyze_phase2.py --figures      # + PNGs
    python 4_analysis/statistical/analyze_phase2.py --stats        # + ANOVA/Tukey/Cohen (requer 5 seeds)

Saídas em results/phase_3/.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
P2 = REPO / "results" / "phase_2" / "base_infix"
OUT = REPO / "results" / "phase_3"

ALGOS = ["best_of_n", "pure_ppo", "pure_grpo", "bon_ppo", "bon_grpo"]
TARGETS = {  # expressão alvo (para complexidade relativa e contexto)
    "nguyen_3": "x**5+x**4+x**3+x**2+x",
    "nguyen_5": "sin(x**2)*cos(x)-1",
    "nguyen_7": "log(x+1)+log(x**2+1)",
    "nguyen_9": "sin(x)+sin(y**2)",
}


def load_runs():
    """Carrega todos os per-seed results_latest.json -> dict[(prob, algo)] = [run, ...]"""
    runs = defaultdict(list)
    for f in sorted(P2.rglob("results_latest.json")):
        parts = f.parts
        i = parts.index("base_infix")
        prob, algo = parts[i + 1], parts[i + 2]
        seed = next((p.split("_")[1] for p in parts if p.startswith("seed_")), "?")
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {f}: {e}")
            continue
        d["_seed"] = int(seed)
        runs[(prob, algo)].append(d)
    # BoN não grava per-seed results_latest: puxar dos aggregates
    for f in sorted(P2.rglob("aggregate_best_of_n_*.json")):
        parts = f.parts
        i = parts.index("base_infix")
        prob, algo = parts[i + 1], parts[i + 2]
        a = json.load(open(f, encoding="utf-8"))
        for k, r in enumerate(a.get("individual_results", [])):
            r["_seed"] = (a.get("seeds") or [k])[k] if k < len(a.get("seeds", [])) else k
            runs[(prob, algo)].append(r)
    return runs


def expr_complexity(expr: str) -> int:
    """Complexidade ~ número de tokens (operadores, funções, variáveis, constantes)."""
    return len(re.findall(r"[a-zA-Z_]+\d*|\*\*|[-+*/()]|\d+\.?\d*", expr or ""))


def dynamics(run: dict) -> dict:
    """Métricas de dinâmica a partir do history por step (None para BoN)."""
    h = run.get("history")
    if not h or not isinstance(h, list) or not isinstance(h[0], dict):
        return {}
    vr = [s.get("valid_rate") for s in h if s.get("valid_rate") is not None]
    kl = [s.get("kl_divergence") for s in h if s.get("kl_divergence") is not None]
    buf = [s.get("buffer_samples_used", 0) for s in h]
    uniq = h[-1].get("total_unique_discovered")
    d = {
        "valid_init": vr[0] if vr else None,
        "valid_min": min(vr) if vr else None,
        "valid_final": np.mean(vr[-10:]) if len(vr) >= 10 else (vr[-1] if vr else None),
        "valid_recovery_step": None,
        "kl_max": max(kl) if kl else None,
        "buffer_used_total": int(np.sum(buf)) if buf else 0,
        "unique_total": uniq,
    }
    if vr and d["valid_min"] is not None and vr[0]:
        # step em que a validade volta a >= 90% do valor inicial após o vale
        imin = int(np.argmin(vr))
        for j in range(imin, len(vr)):
            if vr[j] >= 0.9 * vr[0]:
                d["valid_recovery_step"] = h[j].get("step", j)
                break
    return d


def master_table(runs) -> list[dict]:
    rows = []
    for (prob, algo), rs in sorted(runs.items()):
        r2s = [r.get("best_r2", 0) or 0 for r in rs]
        steps = [r.get("best_step") for r in rs if r.get("best_step") is not None]
        comps = [expr_complexity(r.get("best_expression", "")) for r in rs]
        solved = sum(1 for v in r2s if v > 0.999)
        dyn = [dynamics(r) for r in rs]
        dyn = [d for d in dyn if d]
        row = {
            "problem": prob, "algo": algo, "n_seeds": len(rs),
            "mean_r2": np.mean(r2s), "std_r2": np.std(r2s), "median_r2": np.median(r2s),
            "solved_999": f"{solved}/{len(rs)}",
            "mean_best_step": np.mean(steps) if steps else None,
            "mean_complexity": np.mean(comps) if comps else None,
        }
        if dyn:
            row.update({
                "valid_min": np.mean([d["valid_min"] for d in dyn if d["valid_min"] is not None]),
                "valid_final": np.mean([d["valid_final"] for d in dyn if d["valid_final"] is not None]),
                "kl_max": np.mean([d["kl_max"] for d in dyn if d["kl_max"] is not None]),
                "buffer_used": np.mean([d["buffer_used_total"] for d in dyn]),
                "unique_total": np.mean([d["unique_total"] for d in dyn if d["unique_total"]]),
            })
        rows.append(row)
    return rows


def write_table(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    keys = ["problem", "algo", "n_seeds", "mean_r2", "std_r2", "median_r2", "solved_999",
            "mean_best_step", "mean_complexity", "valid_min", "valid_final", "kl_max",
            "buffer_used", "unique_total"]
    with open(OUT / "master_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    lines = ["| problema | algo | seeds | R² médio±dp | mediana | exatas | step da melhor | complexidade |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['problem']} | {r['algo']} | {r['n_seeds']} "
            f"| {r['mean_r2']:.3f}±{r['std_r2']:.3f} | {r['median_r2']:.3f} | {r['solved_999']} "
            f"| {r['mean_best_step']:.0f}" if r["mean_best_step"] is not None else
            f"| {r['problem']} | {r['algo']} | {r['n_seeds']} "
            f"| {r['mean_r2']:.3f}±{r['std_r2']:.3f} | {r['median_r2']:.3f} | {r['solved_999']} | —")
        lines[-1] += (f" | {r['mean_complexity']:.0f} |" if r.get("mean_complexity") else " | — |")
    (OUT / "master_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ok] {OUT/'master_table.csv'} e .md ({len(rows)} células braço×problema)")


def run_stats(runs):
    """ANOVA two-way aproximada por problema + Tukey + Cohen d nos pares centrais."""
    from scipy import stats as st
    print("\n══ Estatística (problemas com 5 seeds em todos os braços) ══")
    for prob in sorted({p for p, _ in runs}):
        groups = {a: [r.get("best_r2", 0) or 0 for r in runs.get((prob, a), [])] for a in ALGOS}
        if any(len(v) < 5 for v in groups.values()):
            print(f"  {prob}: incompleto ({ {a: len(v) for a, v in groups.items()} }) — pulando")
            continue
        f, p = st.f_oneway(*groups.values())
        print(f"\n  {prob}: ANOVA F={f:.2f} p={p:.4g}")
        res = st.tukey_hsd(*groups.values())
        names = list(groups)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pv = res.pvalue[i, j]
                if pv < 0.05:
                    d = (np.mean(groups[names[i]]) - np.mean(groups[names[j]])) / (
                        np.sqrt((np.var(groups[names[i]]) + np.var(groups[names[j]])) / 2) + 1e-12)
                    print(f"    {names[i]} vs {names[j]}: p={pv:.4f}, Cohen d={d:+.2f}")


def make_figures(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for metric, fname, ylab in [("best_r2", "convergence", "best R²"),
                                ("valid_rate", "validity", "taxa de validade")]:
        probs = sorted({p for p, _ in runs})
        fig, axes = plt.subplots(1, len(probs), figsize=(5 * len(probs), 4), squeeze=False)
        for ax, prob in zip(axes[0], probs):
            for algo in ALGOS:
                rs = runs.get((prob, algo), [])
                curves = []
                for r in rs:
                    h = r.get("history")
                    if h and isinstance(h, list) and isinstance(h[0], dict):
                        curves.append([s.get(metric) or 0 for s in h])
                if not curves:
                    continue
                L = min(len(c) for c in curves)
                arr = np.array([c[:L] for c in curves])
                m = arr.mean(0)
                ax.plot(range(L), m, label=f"{algo} (n={len(curves)})")
                if len(curves) > 1:
                    ax.fill_between(range(L), m - arr.std(0), m + arr.std(0), alpha=0.15)
            ax.set_title(prob); ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / f"{fname}.png", dpi=130)
        plt.close(fig)
        print(f"[ok] {figdir/f'{fname}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    runs = load_runs()
    total = sum(len(v) for v in runs.values())
    print(f"Carregados {total} seed-runs em {len(runs)} células braço×problema")
    rows = master_table(runs)
    write_table(rows)
    if args.stats:
        run_stats(runs)
    if args.figures:
        make_figures(runs)


if __name__ == "__main__":
    main()
