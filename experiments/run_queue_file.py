#!/usr/bin/env python3
"""Roda o daemon de fila sobre um arquivo de fila ESPECÍFICO (modo VM).

Generalização do padrão das filas 1c (legacy/colab_phase1/run_phase1c_A.py):
permite N daemons em paralelo na mesma máquina, cada um com sua fila e seu
lock — sem race no queue.yaml principal.

Uso (no pod, venv ativo, raiz do repo):
    NO_GIT=1 python experiments/run_queue_file.py experiments/queue_phase2_a.yaml --max-hours 24
    NO_GIT=1 python experiments/run_queue_file.py experiments/queue_phase2_b.yaml --max-hours 24
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue_file", type=Path, help="ex.: experiments/queue_phase2_a.yaml")
    ap.add_argument("--max-hours", type=float, default=24.0)
    args = ap.parse_args()

    queue_file = args.queue_file.resolve()
    if not queue_file.exists():
        sys.exit(f"ERRO: fila não encontrada: {queue_file}")

    q = yaml.safe_load(queue_file.read_text(encoding="utf-8"))
    counts = Counter(e.get("status", "pending") for e in q["queue"])
    pending = [e for e in q["queue"] if e.get("status", "pending") == "pending"]
    print(f"Fila {queue_file.name}: {dict(counts)}")
    if not pending:
        print("Nada pendente — encerrando.")
        return
    print(f"Próximo: {pending[0]['id']} — {len(pending)} pendentes\n")

    import experiments.queue_processor as qp
    qp.QUEUE_FILE = queue_file
    qp.LOCK_FILE = queue_file.with_name(f"_running_{queue_file.stem}.lock")
    if qp.LOCK_FILE.exists():
        qp.LOCK_FILE.unlink()
    qp.run_queue_loop(max_hours=args.max_hours)


if __name__ == "__main__":
    main()
