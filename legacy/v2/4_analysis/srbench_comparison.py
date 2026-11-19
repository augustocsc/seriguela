#!/usr/bin/env python3
"""Comparacao preliminar com o SRBench (La Cava et al. 2021).

Calcula NOSSAS metricas no formato do SRBench (grupos Feynman e Strogatz),
usando R² no conjunto de TESTE (75/25 split, protocolo SRBench):
  - solution rate : % de problemas com R²_test > 0.999 (criterio SRBench)
  - accuracy rate : % com R²_test > 0.99
  - mediana/media R²_test
Estratifica Feynman por n_vars (historia de generalizacao H_gen).

RESSALVAS (impressas no rodape):
  - 1 seed (estimativa pontual) vs multi-trial do SRBench
  - C=1, SEM ajuste de constante (handicap; metodos do SRBench ajustam)
  - criterio R²>0.999 e proxy do 'symbolic match via sympy' do SRBench

Os numeros do SRBench (coluna de referencia) sao placeholders a preencher da
fonte (La Cava 2021, fig. de solution rate / leaderboard cavalab/srbench).

Uso: python 4_analysis/srbench_comparison.py
"""
import json, glob
from pathlib import Path
from collections import defaultdict
import statistics as st

REPO = Path(__file__).resolve().parents[1]
RESULT_DIRS = ["results/phase_4_local", "results/phase_4", "results/phase_2"]
BENCH = REPO / "1_data" / "benchmarks"


def nvars(prob):
    for f in glob.glob(str(BENCH / "*" / f"{prob}.meta.json")):
        try:
            return int(json.load(open(f, encoding="utf-8")).get("n_vars", 0))
        except Exception:
            return 0
    for f in glob.glob(str(BENCH / "*" / f"{prob}.meta.txt")):
        import re
        m = re.search(r"n_vars:\s*(\d+)", Path(f).read_text(encoding="utf-8"))
        return int(m.group(1)) if m else 0
    return 0


def load():
    """Por problema: melhor R²_test do campeao bon_ppo (fallback best_r2/train)."""
    rows = {}
    for base in RESULT_DIRS:
        for f in glob.glob(str(REPO / base / "*/bon_ppo/**/results_latest.json"), recursive=True) + \
                 glob.glob(str(REPO / base / "*/bon_ppo/aggregate_*.json")):
            p = Path(f).as_posix().split("/")
            prob = p[p.index("bon_ppo") - 1]
            try:
                d = json.load(open(f, encoding="utf-8"))
                if "individual_results" in d:
                    best = max(d["individual_results"], key=lambda e: e.get("best_r2", -9))
                    tr = d.get("mean_test_r2", best.get("test_r2"))
                    fb = d.get("mean_best_r2", best.get("best_r2"))
                else:
                    tr, fb = d.get("test_r2"), d.get("best_r2")
                r = tr if tr is not None else fb     # prefere test; fallback train
                used_test = tr is not None
                if r is not None and (prob not in rows or r > rows[prob][0]):
                    rows[prob] = (r, used_test)
            except Exception:
                pass
    return rows


def group(prob):
    return "feynman" if prob.startswith("feynman") else \
           "strogatz" if prob.startswith("strogatz") else "nguyen"


def stats(vals):
    if not vals:
        return dict(n=0, sol=0, acc=0, med=0, mean=0)
    return dict(n=len(vals),
                sol=100 * sum(v >= 0.999 for v in vals) / len(vals),
                acc=100 * sum(v >= 0.99 for v in vals) / len(vals),
                med=st.median(vals), mean=sum(vals) / len(vals))


def main():
    rows = load()
    by_group = defaultdict(list)
    test_frac = [u for _, (r, u) in rows.items()]
    for prob, (r, used) in rows.items():
        by_group[group(prob)].append(r)

    print("=" * 64)
    print("NOSSO METODO — campeao (Base Infix, bon_ppo), C=1, 1 seed*")
    print("Metrica: R² no TESTE (proxy do SRBench)")
    print("=" * 64)
    print(f"{'grupo':<12}{'n':>4}{'solution%':>11}{'acc R²>.99%':>13}{'R² mediana':>12}{'R² media':>10}")
    for g in ("feynman", "strogatz", "nguyen"):
        s = stats(by_group[g])
        if s["n"]:
            print(f"{g:<12}{s['n']:>4}{s['sol']:>10.1f}%{s['acc']:>12.1f}%{s['med']:>12.3f}{s['mean']:>10.3f}")
    gt = by_group["feynman"] + by_group["strogatz"]
    s = stats(gt)
    print(f"{'GT (Fey+Str)':<12}{s['n']:>4}{s['sol']:>10.1f}%{s['acc']:>12.1f}%{s['med']:>12.3f}{s['mean']:>10.3f}")

    print("\n--- Feynman estratificado por n_vars (generalizacao H_gen) ---")
    byv = defaultdict(list)
    for prob, (r, _) in rows.items():
        if prob.startswith("feynman"):
            byv[nvars(prob)].append(r)
    print(f"{'n_vars':<8}{'n':>4}{'solution%':>11}{'R² mediana':>12}")
    for v in sorted(byv):
        s = stats(byv[v])
        print(f"{v:<8}{s['n']:>4}{s['sol']:>10.1f}%{s['med']:>12.3f}")

    used_test = sum(test_frac)
    print(f"\n* {used_test}/{len(rows)} usam R²_test; resto usa R²_train (sem test salvo).")
    print("RESSALVAS: 1 seed (pontual, nao taxa); C=1 SEM ajuste de constante")
    print("(handicap — SRBench ajusta); R²>0.999 e proxy do symbolic-match deles.")
    print("\nReferencia SRBench (La Cava 2021) — PREENCHER da fonte:")
    print("  solution rate Feynman (noise 0): lideres gplearn/AFP/AIFeynman/DSR")
    print("  melhor R²: Operon/SBP-GP/GP-GOMEA. Ver fig. solution-rate + accuracy.")


if __name__ == "__main__":
    main()
