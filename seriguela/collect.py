"""Uma tabela a partir de ``results/v3``: uma linha por run, sem preencher lacunas.

    python -m seriguela.collect results/v3 --csv tabela.csv

Recusa misturar runs de commits diferentes (a menos de ``--allow_mixed``): uma
tabela de campanha vem de UM código. Run que falta aparece como falta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .trainer import SOLVED_R2, validate


class MixedProvenanceError(RuntimeError):
    """Os runs foram produzidos por mais de um commit."""


def load(root: Path, allow_mixed: bool = False) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(Path(root).rglob("seed_*.json")):
        if path.name.endswith(".failed.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        validate(result)
        best, final = result["best"] or {}, result["final_policy"]
        test_r2 = best.get("test_r2")
        rows.append({
            "problem": result["problem"], "arm": result["arm"], "seed": result["seed"],
            "status": result["status"], "commit": result["provenance"]["commit"],
            "dirty": result["provenance"]["dirty"], "config_hash": result["provenance"]["config_hash"],
            "samples": result["counters"]["samples"], "seconds": result["counters"]["seconds"],
            "expression": best.get("expression"), "train_r2": best.get("train_r2"), "test_r2": test_r2,
            "solved_accuracy": bool(test_r2 is not None and test_r2 >= SOLVED_R2),
            "solved_symbolic": best.get("symbolic_match"),
            "final_solved_frac": final["solved_frac"], "final_valid_rate": final["valid_rate"],
            "kl_from_prior": final["kl_from_prior_per_token"], "path": str(path),
        })
    frame = pd.DataFrame(rows)
    if len(frame) and not allow_mixed:
        if frame["commit"].nunique() > 1:
            raise MixedProvenanceError(f"commits misturados: {sorted(frame['commit'].unique())}")
        if frame["dirty"].any():
            raise MixedProvenanceError("há runs feitos com a árvore git suja")
    return frame


def failures(root: Path) -> List[Path]:
    return sorted(Path(root).rglob("seed_*.failed.json"))


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Por problema e braço: quantas seeds, quantas resolvidas em cada critério."""
    grouped = frame.groupby(["problem", "arm"])
    return grouped.agg(
        seeds=("seed", "nunique"),
        collapsed=("status", lambda s: int((s != "ok").sum())),
        symbolic=("solved_symbolic", lambda s: int((s == True).sum())),   # noqa: E712 (None conta como não)
        accuracy=("solved_accuracy", "sum"),
        median_test_r2=("test_r2", "median"),
        median_train_r2=("train_r2", "median"),
    ).reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--csv")
    parser.add_argument("--allow_mixed", action="store_true")
    args = parser.parse_args(argv)
    frame = load(Path(args.root), args.allow_mixed)
    failed = failures(Path(args.root))
    if len(frame):
        print(summary(frame).to_string(index=False))
    print(f"\n{len(frame)} runs válidos; {len(failed)} interrompidos por tripwire")
    for path in failed:
        print(f"  FALHOU: {path}")
    if args.csv:
        frame.to_csv(args.csv, index=False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
