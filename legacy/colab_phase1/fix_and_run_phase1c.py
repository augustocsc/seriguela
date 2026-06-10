"""
Fix do erro de torchao e reinício da Phase 1c no Colab.
Cole e execute numa célula do Colab.

Problema: peft==0.15.1 exige torchao>=0.16.0, mas Colab tem 0.10.0.
Fix: desinstalar torchao (não usamos quantização torchao — peft funciona sem ele).
"""

import subprocess, sys, os
from pathlib import Path

REPO = Path("/content/seriguela")

# ── 1. Fix: remover torchao ────────────────────────────────────────────────────
print("=== Fix: removendo torchao (não é necessário para nossos experimentos) ===")
r = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "torchao", "-y", "-q"],
    capture_output=True, text=True
)
if r.returncode == 0:
    print("torchao removido com sucesso.")
else:
    # torchao pode não estar instalado de forma que pip reconheça — tudo bem
    print(f"(torchao já ausente ou não removível via pip — seguindo: {r.stderr.strip()[:120]})")

# Verificar que peft agora carrega sem erro
print("\n=== Verificando peft + gpt2_large_infix_682k ===")
try:
    import importlib
    import peft
    importlib.reload(peft)          # recarrega peft sem torchao no path
    print(f"peft {peft.__version__} importado com sucesso.")
except Exception as e:
    print(f"ERRO ao importar peft: {e}")
    print("Tentando alternativa: downgrade peft para 0.13.2 ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "peft==0.13.2", "-q"],
        check=True
    )
    print("peft 0.13.2 instalado. Reinicie o kernel e execute de novo.")
    raise SystemExit("Reinicie o kernel do Colab e execute esta célula novamente.")

# ── 2. Remover lock se existir ─────────────────────────────────────────────────
lock = REPO / "experiments" / "_running.lock"
if lock.exists():
    lock.unlink()
    print(f"\nLock removido: {lock}")

# ── 3. Atualizar repo (pode ter patches novos) ─────────────────────────────────
print("\n=== git pull ===")
r = subprocess.run(["git", "pull", "--ff-only"], cwd=str(REPO), capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())

# ── 4. Checar experimentos pendentes ──────────────────────────────────────────
import yaml
with open(REPO / "experiments" / "queue.yaml") as f:
    q = yaml.safe_load(f)
exps = q["queue"]
from collections import Counter
counts = Counter(e.get("status", "pending") for e in exps)
print(f"\nFila: {dict(counts)}")
pending = [e for e in exps if e.get("status", "pending") == "pending"]
print(f"Próximo: {pending[0]['id'] if pending else '—'}")

if not pending:
    print("\nNenhum experimento pendente. Fila completa!")
    raise SystemExit(0)

# ── 5. Rodar daemon ────────────────────────────────────────────────────────────
print("\n=== Iniciando daemon Phase 1c (Large 774M, 122 problemas) ===")
print("Pressione o botão de parada da célula para interromper com segurança.\n")

sys.path.insert(0, str(REPO))
from experiments.queue_processor import run_queue_loop
run_queue_loop(max_hours=11.0)
