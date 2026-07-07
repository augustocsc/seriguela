#!/usr/bin/env python3
"""Comparacao do nosso campeao vs o leaderboard OFICIAL do SRBench (La Cava 2021).

Usa os resultados crus publicados em cavalab/srbench
(results/ground-truth_results.feather, baixado via LFS media) e computa, em
ruido 0, por metodo e por grupo (Feynman/Strogatz):
  - solution rate : SRBench usa symbolic_solution (sympy); o NOSSO usa R²_test>0.999 (proxy)
  - acc R²>0.99   : fracao com R²_test>0.99
  - R² mediana    : mediana do R²_test

Nosso lado vem de results/phase_{2,4,4_local}/.../bon_ppo (campeao, C=1).

RESSALVAS: 1 seed (pontual); C=1 SEM ajuste de constante (SRBench ajusta ->
ver coluna R²>0.99); nosso 'solution' e proxy de R², nao o symbolic-match sympy
(rigor pleno exige rodar experiment/assess_symbolic_model.py do SRBench nas
nossas expressoes).

Uso: python 4_analysis/srbench_leaderboard.py
Saida: results/srbench_comparison.md + stdout.
"""
import glob
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FEATHER = REPO / "1_data" / "benchmarks" / "srbench_ground-truth_results.feather"
URL = ("https://media.githubusercontent.com/media/cavalab/srbench/master/"
       "results/ground-truth_results.feather")
OURS_DIRS = ["results/phase_4_local", "results/phase_4", "results/phase_2"]


def ensure_feather():
    if FEATHER.exists() and FEATHER.stat().st_size > 1_000_000:
        return
    import urllib.request
    print(f"baixando SRBench feather (~79MB) de {URL} ...")
    FEATHER.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, FEATHER)


def our_results():
    rows = {}
    for base in OURS_DIRS:
        for f in glob.glob(str(REPO / base / "*/bon_ppo/**/results_latest.json"), recursive=True) + \
                 glob.glob(str(REPO / base / "*/bon_ppo/aggregate_*.json")):
            p = Path(f).as_posix().split("/")
            prob = p[p.index("bon_ppo") - 1]
            try:
                d = json.load(open(f, encoding="utf-8"))
                if "individual_results" in d:
                    b = max(d["individual_results"], key=lambda e: e.get("best_r2", -9))
                    r = d.get("mean_test_r2", b.get("test_r2")) or d.get("mean_best_r2", b.get("best_r2"))
                else:
                    r = d.get("test_r2") or d.get("best_r2")
                if r is not None and (prob not in rows or r > rows[prob]):
                    rows[prob] = r
            except Exception:
                pass
    return rows


def ours_row(rows, pref):
    v = [r for p, r in rows.items() if p.startswith(pref)]
    if not v:
        return None
    return (100 * sum(x >= 0.999 for x in v) / len(v),
            100 * sum(x >= 0.99 for x in v) / len(v),
            sorted(v)[len(v) // 2], len(v))


def main():
    ensure_feather()
    df = pd.read_feather(FEATHER)
    df = df[df["target_noise"] == 0.0]
    rows = our_results()

    out = ["# Nosso campeao (Base Infix, bon_ppo, C=1, 1 seed) vs SRBench leaderboard",
           "", "Ruido 0. `solution`: SRBench=symbolic_solution (sympy); nosso=R²>0.999 (proxy).", ""]
    for grp, pref in [("Feynman", "feynman"), ("Strogatz", "strogatz")]:
        g = df[df["data_group"] == grp]
        tab = []
        o = ours_row(rows, pref)
        if o:
            tab.append(("**NOSSO (bon_ppo)**", o[0], o[1], o[2]))
        for alg, sub in g.groupby("algorithm"):
            tab.append((alg, 100 * sub["symbolic_solution"].mean(),
                        100 * (sub["r2_test"] > 0.99).mean(), sub["r2_test"].median()))
        tab.sort(key=lambda x: -x[1])
        out += [f"## {grp} (ruido 0)", "",
                "| metodo | solution % | R²>0.99 % | R² mediana |",
                "|---|---|---|---|"]
        for name, sol, acc, med in tab:
            out.append(f"| {name} | {sol:.1f} | {acc:.1f} | {med:.3f} |")
        out.append("")
    (REPO / "results" / "srbench_comparison.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print("[salvo] results/srbench_comparison.md")


if __name__ == "__main__":
    main()
