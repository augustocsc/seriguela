#!/usr/bin/env python3
"""Tabela-mae da Fase 4 — todos os desafios do benchmark x status/score/equacao.

Enumera nguyen + strogatz + feynman de 1_data/benchmarks/ e cruza com os
resultados do campeao (bon_ppo) coletados em:
  - results/phase_4_local   (nguyen na 3050)
  - results/phase_4         (strogatz + feynman no RunPod)
  - results/phase_2/...     (nguyen 3/5/7/9 ja vieram da Fase 2)

Equacao-objetivo: nguyen vem de *.meta.txt (campo equation); strogatz/feynman
sao extraidos da descricao PMLB em *.meta.json (heuristica de linha com '=').

Saida: tabela markdown no stdout + results/mother_table.md + results/mother_table.csv

Uso:
    python 4_analysis/build_mother_table.py
    python 4_analysis/build_mother_table.py --sort score     # ordena por R²
    python 4_analysis/build_mother_table.py --only done       # so concluidos
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "1_data" / "benchmarks"
RESULT_DIRS = ["results/phase_4_local", "results/phase_4", "results/phase_2"]
SOLVE_THRESHOLD = 0.999  # criterio SRBench


def category(name):
    if name.startswith("nguyen"):
        return "nguyen"
    if name.startswith("strogatz"):
        return "strogatz"
    return "feynman"


def extract_formula(description):
    """Pega a 1a linha tipo 'lhs = rhs' da descricao PMLB (heuristica)."""
    if not description:
        return ""
    for line in description.splitlines():
        s = line.strip()
        if "=" in s and re.search(r"[a-zA-Z].*=.*[a-zA-Z0-9]", s) \
                and not re.search(r"\brange|\bvariable|\bsimulated|units\b", s, re.I):
            return re.sub(r"\s+", " ", s)[:80]
    return ""


def enumerate_problems():
    probs = {}
    for f in glob.glob(str(BENCH / "nguyen" / "*.meta.txt")):
        meta = dict(re.findall(r"^(\w+):\s*(.+)$", Path(f).read_text(encoding="utf-8"), re.M))
        name = meta.get("name", Path(f).stem)
        probs[name] = {"cat": "nguyen", "n_vars": meta.get("n_vars", "?"),
                       "target": meta.get("equation", "")}
    for sub in ("strogatz", "feynman"):
        for f in glob.glob(str(BENCH / sub / "*.meta.json")):
            d = json.load(open(f, encoding="utf-8"))
            name = d.get("name", Path(f).stem.replace(".meta", ""))
            desc = (d.get("pmlb_metadata") or {}).get("description", "")
            probs[name] = {"cat": sub, "n_vars": str(d.get("n_vars", "?")),
                           "target": extract_formula(desc)}
    return probs


def find_result(problem):
    """Procura o melhor resultado bon_ppo do problema entre os RESULT_DIRS.

    Prefere aggregate_*.json (multi-seed: media + solution_rate); senao usa o
    results_latest.json (single-seed). Retorna dict normalizado ou None.
    """
    # aggregate primeiro (multi-seed)
    for base in RESULT_DIRS:
        for agg in glob.glob(str(REPO / base / f"**/{problem}/bon_ppo/aggregate_*.json"),
                             recursive=True):
            try:
                d = json.load(open(agg, encoding="utf-8"))
                ind = (d.get("individual_results") or [{}])
                best = max(ind, key=lambda r: r.get("best_r2", -9)) if ind else {}
                return {
                    "status": "done", "seeds": d.get("n_solved") is not None and len(d.get("seeds", [])) or len(ind),
                    "best_r2": d.get("mean_best_r2", best.get("best_r2")),
                    "test_r2": d.get("mean_test_r2", best.get("test_r2")),
                    "expr": best.get("best_expression", ""),
                    "best_step": best.get("best_step"),
                    "sol_rate": d.get("solution_rate"),
                }
            except Exception:
                pass
    # single results_latest
    for base in RESULT_DIRS:
        for rf in glob.glob(str(REPO / base / f"**/{problem}/bon_ppo/**/results_latest.json"),
                            recursive=True):
            try:
                d = json.load(open(rf, encoding="utf-8"))
                return {"status": "done", "seeds": 1, "best_r2": d.get("best_r2"),
                        "test_r2": d.get("test_r2"), "expr": d.get("best_expression", ""),
                        "best_step": d.get("best_step"), "sol_rate": None}
            except Exception:
                pass
    return None


def build_rows():
    probs = enumerate_problems()
    rows = []
    for name, info in probs.items():
        r = find_result(name)
        if r:
            best = r["best_r2"]
            solved = "yes" if (best is not None and best >= SOLVE_THRESHOLD) else "no"
            row = {**info, "problem": name, "status": "done", "seeds": r["seeds"],
                   "best_r2": round(best, 4) if best is not None else None,
                   "test_r2": round(r["test_r2"], 4) if r["test_r2"] is not None else None,
                   "solved": solved, "found": (r["expr"] or "")[:60], "best_step": r["best_step"]}
        else:
            row = {**info, "problem": name, "status": "pending", "seeds": 0,
                   "best_r2": None, "test_r2": None, "solved": "-", "found": "", "best_step": None}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sort", choices=["name", "score", "cat"], default="cat")
    ap.add_argument("--only", choices=["all", "done", "pending"], default="all")
    ap.add_argument("--csv", default="results/mother_table.csv")
    ap.add_argument("--md", default="results/mother_table.md")
    args = ap.parse_args()

    rows = build_rows()
    if args.only != "all":
        rows = [r for r in rows if r["status"] == args.only]
    catord = {"nguyen": 0, "strogatz": 1, "feynman": 2}
    if args.sort == "score":
        rows.sort(key=lambda r: (r["best_r2"] is None, -(r["best_r2"] or 0)))
    elif args.sort == "name":
        rows.sort(key=lambda r: r["problem"])
    else:
        rows.sort(key=lambda r: (catord[r["cat"]], -(r["best_r2"] or -1)))

    cols = ["problem", "cat", "n_vars", "status", "seeds", "best_r2", "test_r2",
            "solved", "found", "target", "best_step"]
    hdr = ["desafio", "cat", "vars", "status", "seeds", "best R²", "test R²",
           "resolv.", "equacao encontrada", "equacao objetivo", "step"]

    # markdown
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(
            str(r.get(c)) if r.get(c) is not None else "—" for c in cols) + " |")

    done = [r for r in rows if r["status"] == "done"]
    solved = [r for r in done if r["solved"] == "yes"]
    scored = [r["best_r2"] for r in done if r["best_r2"] is not None]
    summary = (f"\n**{len(rows)} desafios** | concluidos {len(done)} | "
               f"resolvidos (R²>{SOLVE_THRESHOLD}) {len(solved)} | "
               f"best R² medio dos concluidos "
               f"{round(sum(scored)/len(scored),3) if scored else 0}")

    out_md = "\n".join(md) + "\n" + summary + "\n"
    (REPO / args.md).write_text(out_md, encoding="utf-8")
    with open(REPO / args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    print(out_md)
    print(f"[salvo] {args.md} e {args.csv}")


if __name__ == "__main__":
    main()
