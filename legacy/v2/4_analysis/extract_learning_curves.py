#!/usr/bin/env python3
"""Extrai as curvas de aprendizado da Fase 4 (local 3050 + RunPod) para visualizacao.

Sinais por desafio (de history[]):
  best  = best_r2 acumulado (monotonico — "ja achou solucao?")
  p90   = fresh_p90_r2 — qualidade do top-10% das amostras FRESCAS (sinal real
          de aprendizado da politica; sobe se o modelo melhora)
  vrate = fresh_valid_rate — fracao de expressoes validas (estabilidade)

Saida: results/_learning_curves.json (alimenta o widget).
"""
import json, glob, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def clip(v, lo, hi):
    try:
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return lo if v < 0 else hi
        return max(lo, min(hi, v))
    except Exception:
        return lo


def load_curve(f):
    d = json.load(open(f, encoding="utf-8"))
    h = d.get("history") or []
    if not h:
        return None
    steps = [p.get("step", i) for i, p in enumerate(h)]
    best = [clip(p.get("best_r2", 0), -1, 1) for p in h]
    p90 = [clip(p.get("fresh_p90_r2", p.get("fresh_median_r2", 0)), -1, 1) for p in h]
    vr = [clip(p.get("fresh_valid_rate", p.get("valid_rate", 0)), 0, 1) for p in h]
    n = len(h)
    idx = list(range(0, n, max(1, n // 40)))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return {
        "steps": [steps[i] for i in idx],
        "best": [round(best[i], 4) for i in idx],
        "p90": [round(p90[i], 4) for i in idx],
        "vrate": [round(vr[i], 3) for i in idx],
        "final_best": round(best[-1], 4), "best_step": d.get("best_step"),
        "final_p90": round(p90[-1], 4), "init_p90": round(p90[0], 4),
        "final_vrate": round(vr[-1], 3), "init_vrate": round(vr[0], 3),
        "expr": (d.get("best_expression") or "")[:50],
    }


def main():
    probs = {}
    for base in ["results/phase_4_local", "results/phase_4"]:
        for f in glob.glob(str(REPO / base / "*/bon_ppo/**/results_latest.json"),
                           recursive=True):
            parts = Path(f).as_posix().split("/")
            prob = parts[parts.index("bon_ppo") - 1]
            try:
                c = load_curve(f)
                if c:
                    probs[prob] = c
            except Exception as e:
                print("SKIP", prob, e)

    def cat(p):
        return 0 if p.startswith("nguyen") else (1 if p.startswith("strogatz") else 2)

    order = sorted(probs.keys(), key=lambda p: (cat(p), -probs[p]["final_best"]))
    json.dump({"order": order, "probs": probs},
              open(REPO / "results/_learning_curves.json", "w"), indent=0)

    print(f"Extraidos {len(probs)} desafios")
    print(f"{'problema':<28}{'best':>7}{'@step':>7}  {'p90 ini->fim':>14}  {'valid ini->fim':>14}")
    for p in order:
        c = probs[p]
        print(f"{p:<28}{c['final_best']:>7.3f}{str(c['best_step']):>7}  "
              f"{c['init_p90']:>6.2f}->{c['final_p90']:<6.2f}  "
              f"{c['init_vrate']:>6.2f}->{c['final_vrate']:<6.2f}")


if __name__ == "__main__":
    main()
