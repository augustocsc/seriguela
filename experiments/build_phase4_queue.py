#!/usr/bin/env python3
"""Gera a fila da Fase 4.2 (THESIS_PLAN §4.2): campeão RL em TODOS os 122 problemas.

Campeão fixado nas Fases 2/3: **(Base Infix, bon_ppo)** — o Base venceu o Large
na seleção da Fase 1 (zero-shot mean_best_r2 -0.39 vs -1.17; ver phase_1b/1c).
Config idêntica à da Fase 2 (build_phase2_queue): r2_clipped / gradient /
cosine_annealing / standard, batch 1024, patience alto, tuning PPO
(lr 1e-6, 2 épocas, max_kl 0.02), buffer 0.2, C=1.

Escopo desta fila: o VOLUME que roda no RunPod, complementando a 3050 local:
  - strogatz (14) + feynman (~101)  — lidos dos arquivos em 1_data/benchmarks/
  - nguyen FICA DE FORA (3/5/7/9 já vêm da Fase 2; 1/2/4/6/8/10/11/12 rodam local)
  - 3 seeds: 42 123 456

Saída: experiments/queue_phase4.yaml (+ --split N -> queue_phase4_{a,b,..}.yaml
para N daemons concorrentes via run_queue_file.py).

Uso:
    python experiments/build_phase4_queue.py                 # preview
    python experiments/build_phase4_queue.py --write --split 3
"""
import argparse
import datetime
import string
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "1_data" / "benchmarks"
QUEUE_FILE = REPO_ROOT / "experiments" / "queue_phase4.yaml"

MODEL = "augustocsc/gpt2_base_infix_682k"
SEEDS = [42, 123, 456]


def discover_problems() -> list:
    """strogatz (CSV) + feynman (meta.json), ordenados — nguyen fica de fora."""
    strogatz = sorted(p.stem for p in (BENCH / "strogatz").glob("*.csv"))
    feynman = sorted(p.name[:-len(".meta.json")]
                     for p in (BENCH / "feynman").glob("*.meta.json"))
    return strogatz + feynman


def build_entries(problems: list, max_steps: int, batch_size: int) -> list:
    today = datetime.date.today().isoformat()
    entries = []
    for problem in problems:
        args = {
            "algorithm": "bon_ppo",
            "model": MODEL,
            "problem": problem,
            "seeds": SEEDS,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "reward": "r2_clipped",
            "penalty": "gradient",
            "temperature": "cosine_annealing",
            "prompt_type": "standard",
            "patience": 100000,
            "no_wandb": True,
            "buffer_ratio": 0.2,
            # tuning PPO validado (bon_ppo é variante PPO); ver build_phase2_queue
            "learning_rate": 1e-6,
            "ppo_epochs": 2,
            "max_kl": 0.02,
        }
        entries.append({
            "id": f"phase4_base_infix_bon_ppo_{problem}",
            "phase": 4,
            "status": "pending",
            "created": today,
            "args": args,
            "output_dir": f"results/phase_4/{problem}/bon_ppo/",
            # 3 seeds x ~200 steps; bon_ppo concorrente ~10-38 s/step (piloto)
            "max_hours": 4.0,
            "estimated_minutes": 150,
        })
    return entries


def write_split(entries: list, n: int, base_meta: dict):
    for i in range(n):
        part = entries[i::n]
        pf = QUEUE_FILE.with_name(f"queue_phase4_{string.ascii_lowercase[i]}.yaml")
        pf.write_text(yaml.safe_dump(
            {"defaults": base_meta.get("defaults", {}),
             "meta": base_meta.get("meta", {}), "queue": part},
            sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"  split {pf.name}: {len(part)} entradas")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--split", type=int, default=0, metavar="N",
                    help="escreve N filas paralelas para N daemons concorrentes")
    args = ap.parse_args()

    problems = discover_problems()
    entries = build_entries(problems, args.max_steps, args.batch_size)
    strogatz = [p for p in problems if p.startswith("strogatz")]
    feynman = [p for p in problems if p.startswith("feynman")]

    print(f"Fase 4.2 — campeão (Base Infix, bon_ppo) em {len(problems)} problemas "
          f"({len(strogatz)} strogatz + {len(feynman)} feynman) x {len(SEEDS)} seeds "
          f"= {len(problems)*len(SEEDS)} seed-runs")
    print(f"nguyen FORA (3/5/7/9 da Fase 2; resto na 3050 local)")
    for e in entries[:3]:
        print(f"  + {e['id']}")
    print(f"  ... (+{len(entries)-3} entradas)")

    if not args.write:
        print("\nDry-run. Re-run com --write [--split N].")
        return

    meta = {"defaults": {"max_hours": 4.0, "retry_on_oom": True},
            "meta": {"phase": 4, "champion": "base_infix/bon_ppo",
                     "created": datetime.date.today().isoformat()}}
    QUEUE_FILE.write_text(yaml.safe_dump(
        {**meta, "queue": entries}, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    print(f"\nEscrito: {QUEUE_FILE.name} ({len(entries)} entradas)")
    if args.split:
        write_split(entries, args.split, meta)
    print("Validar com: python experiments/test_smoke.py")


if __name__ == "__main__":
    main()
