"""
NOTEBOOK B — Phase 1c (experimentos 2/2: 61 problemas)

Cole e execute numa célula do Colab. O script detecta automaticamente
se é a primeira vez (torchao presente) ou a segunda (kernel limpo).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÉLULA 0 — setup (só uma vez):
    !git clone https://github.com/augustocsc/seriguela.git /content/seriguela
    !pip install transformers==4.51.3 peft==0.15.1 accelerate==1.6.0 datasets==3.5.0 trl==0.16.1 sympy==1.13.1 pyyaml -q

CÉLULA 1 — fix (vai reiniciar o kernel):
    exec(open("/content/seriguela/experiments/run_phase1c_B.py").read())

CÉLULA 2 — daemon (execute após o kernel reiniciar):
    exec(open("/content/seriguela/experiments/run_phase1c_B.py").read())
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess, sys, os
from pathlib import Path

REPO       = Path("/content/seriguela")
QUEUE_FILE = REPO / "experiments" / "queue_1c_b.yaml"
LOCK_FILE  = REPO / "experiments" / "_running_b.lock"

# ── Detectar torchao ──────────────────────────────────────────────────────────
def torchao_present() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("torchao") is not None
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════════════════════
# PASSO 1: torchao presente → remover e reiniciar kernel
# ══════════════════════════════════════════════════════════════════════════════
if torchao_present():
    print("=" * 60)
    print("NOTEBOOK B — PASSO 1/2: Removendo torchao + reiniciando kernel")
    print("=" * 60)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "torchao", "-y"],
        capture_output=True, text=True
    )
    print("OK:", r.stdout.strip() or "torchao removido")
    print("\nAguarde o kernel reiniciar e então execute esta célula novamente.")

    try:
        from google.colab.output import eval_js
        eval_js("google.colab.kernel.invokeFunction('notebook.RestartKernel', [], {})")
    except Exception:
        import signal
        os.kill(os.getpid(), signal.SIGKILL)
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════════
# PASSO 2: kernel limpo → configurar e rodar daemon
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("NOTEBOOK B — PASSO 2/2: Iniciando daemon (61 experimentos)")
print("=" * 60)

# Verificar peft
try:
    import peft
    print(f"peft {peft.__version__} OK")
except Exception as e:
    print(f"ERRO peft: {e}")
    raise

# Configurar git identity (necessário para commits funcionarem)
subprocess.run(["git", "config", "--global", "user.email", "colab@experiment.local"], cwd=str(REPO))
subprocess.run(["git", "config", "--global", "user.name", "Colab Runner"], cwd=str(REPO))
print("Git identity: OK")

# Git pull
r = subprocess.run(["git", "pull", "--ff-only"], cwd=str(REPO), capture_output=True, text=True)
print("git pull:", r.stdout.strip() or r.stderr.strip() or "up to date")

# Remover lock se existir
if LOCK_FILE.exists():
    LOCK_FILE.unlink()
    print("Lock removido.")

# Status da fila B
import yaml
with open(QUEUE_FILE) as f:
    q = yaml.safe_load(f)
exps = q["queue"]
from collections import Counter
counts = Counter(e.get("status", "pending") for e in exps)
pending = [e for e in exps if e.get("status", "pending") == "pending"]
print(f"\nFila B: {dict(counts)}")
if not pending:
    print("Todos os experimentos da fila B já concluídos!")
    raise SystemExit(0)
print(f"Próximo: {pending[0]['id']}")
print(f"Iniciando {len(pending)} experimentos...\n")

# Rodar daemon com a fila B e lock B
sys.path.insert(0, str(REPO))
import experiments.queue_processor as qp
qp.QUEUE_FILE = QUEUE_FILE
qp.LOCK_FILE  = LOCK_FILE
qp.run_queue_loop(max_hours=11.0)
