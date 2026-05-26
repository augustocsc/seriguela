"""
NOTEBOOK B — Phase 1c (61 experimentos)

Setup (célula 0):
    !GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/augustocsc/seriguela.git /content/seriguela
    !pip install transformers==4.51.3 peft==0.15.1 accelerate==1.6.0 datasets==3.5.0 trl==0.16.1 sympy==1.13.1 pyyaml -q
    !pip uninstall torchao -y
    # Reinicie o kernel: Runtime → Restart runtime

Rodar (célula 1):
    !python /content/seriguela/experiments/run_phase1c_B.py
"""

import sys, subprocess
from pathlib import Path
from collections import Counter

REPO = Path("/content/seriguela")
sys.path.insert(0, str(REPO))

# Git identity para commits funcionarem
subprocess.run(["git", "config", "--global", "user.email", "colab@experiment.local"], check=False)
subprocess.run(["git", "config", "--global", "user.name", "Colab Runner"], check=False)

import peft
print(f"peft {peft.__version__} OK")

import yaml
QUEUE_FILE = REPO / "experiments" / "queue_1c_b.yaml"
LOCK_FILE  = REPO / "experiments" / "_running_b.lock"

with open(QUEUE_FILE) as f:
    q = yaml.safe_load(f)

counts = Counter(e.get("status", "pending") for e in q["queue"])
pending = [e for e in q["queue"] if e.get("status", "pending") == "pending"]
print(f"Fila B: {dict(counts)}")
if not pending:
    print("Todos os experimentos da fila B já concluídos!")
    sys.exit(0)
print(f"Próximo: {pending[0]['id']} — iniciando {len(pending)} experimentos...\n")

if LOCK_FILE.exists():
    LOCK_FILE.unlink()

import experiments.queue_processor as qp
qp.QUEUE_FILE = QUEUE_FILE
qp.LOCK_FILE  = LOCK_FILE
qp.run_queue_loop(max_hours=11.0)
