"""
experiments/queue_processor.py
Daemon que processa a fila de experimentos em experiments/queue.yaml.

Uso no Colab (via seriguela_runner.ipynb):
    from experiments.queue_processor import run_queue_loop
    run_queue_loop(max_hours=11)

Comportamentos críticos:
- Idempotência: se aggregate JSON já existe em output_dir, marca `done` e pula
- Backoff em falha: marca `failed` com error message, vai pro próximo
- Heartbeat: atualiza experiments/heartbeat.json a cada 5 min com timestamp + experiment atual
- Commit batch: a cada 5 experimentos `done`, faz UM commit no git (não 5)
- Resume: ao iniciar, pula todos os `done/failed/skipped`, começa no próximo `pending`
"""

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import yaml


# ─── Configurações ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent  # raiz do repo seriguela
QUEUE_FILE = REPO_ROOT / "experiments" / "queue.yaml"
HEARTBEAT_FILE = REPO_ROOT / "experiments" / "heartbeat.json"
LOCK_FILE = REPO_ROOT / "experiments" / "_running.lock"

HEARTBEAT_INTERVAL_SEC = 300  # 5 minutos
BATCH_COMMIT_EVERY = 5  # commit a cada N experimentos done
GIT_PULL_INTERVAL_SEC = 60  # git pull a cada 1 min


# ─── Utilitários ──────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def git_pull():
    """Pull das últimas mudanças do queue.yaml (Claude pode ter adicionado mais jobs)."""
    try:
        result = subprocess.run(
            ["git", "pull", "--rebase", "--autostash"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log(f"git pull OK: {result.stdout.strip()[:100]}")
        else:
            log(f"git pull WARN: {result.stderr.strip()[:200]}")
    except Exception as e:
        log(f"git pull FAILED (non-fatal): {e}")


def git_commit_batch(message: str):
    """Commit das mudanças de queue.yaml + resultados gerados."""
    try:
        subprocess.run(["git", "add", "experiments/queue.yaml", "experiments/heartbeat.json"],
                       cwd=str(REPO_ROOT), check=False)
        # Adiciona resultados novos (apenas os que mudaram)
        subprocess.run(["git", "add", "results/"], cwd=str(REPO_ROOT), check=False)

        result = subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            log(f"git commit OK: {message}")
            # Tenta push (pode falhar se sem internet — não fatal)
            push = subprocess.run(
                ["git", "push"],
                cwd=str(REPO_ROOT),
                capture_output=True, text=True, timeout=30
            )
            if push.returncode == 0:
                log("git push OK")
            else:
                log(f"git push WARN (continuando): {push.stderr.strip()[:100]}")
                # Push falhou — salva resultados no W&B Artifacts como backup
                wandb_backup_results()
        else:
            log(f"git commit WARN: {result.stderr.strip()[:200]}")
    except Exception as e:
        log(f"git commit FAILED (non-fatal): {e}")


def wandb_backup_results():
    """Salva results/ como W&B Artifact quando o git push falha.

    Só executa se WANDB_API_KEY estiver disponível. Não fatal.
    """
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            log("wandb backup: WANDB_API_KEY não encontrado, pulando")
            return

        results_dir = REPO_ROOT / "results"
        jsons = list(results_dir.rglob("aggregate_*.json"))
        if not jsons:
            log("wandb backup: nenhum JSON de resultado encontrado")
            return

        log(f"wandb backup: enviando {len(jsons)} JSONs para W&B Artifacts...")
        run = wandb.init(
            project="seriguela",
            entity="symbolic-gression",
            job_type="results-backup",
            name=f"backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            reinit=True,
        )
        artifact = wandb.Artifact(
            "experiment-results",
            type="dataset",
            description=f"Auto-backup — {len(jsons)} JSONs (git push falhou)",
        )
        artifact.add_dir(str(results_dir), name="results")
        run.log_artifact(artifact)
        run.finish()
        log(f"wandb backup OK: {len(jsons)} arquivos salvos em W&B Artifacts")
    except Exception as e:
        log(f"wandb backup FAILED (non-fatal): {e}")


def write_heartbeat(current_exp_id: Optional[str] = None, done_count: int = 0):
    """Escreve heartbeat.json no Drive e no repo (para monitoramento remoto)."""
    heartbeat = {
        "timestamp": now_iso(),
        "current_experiment": current_exp_id,
        "experiments_done_this_session": done_count,
        "queue_file": str(QUEUE_FILE),
        "pid": os.getpid(),
    }
    with open(HEARTBEAT_FILE, "w") as f:
        json.dump(heartbeat, f, indent=2)

    # Copia para Drive se disponível
    drive_heartbeat = Path("/content/drive/MyDrive/seriguela_results/heartbeat.json")
    if drive_heartbeat.parent.exists():
        drive_heartbeat.write_text(json.dumps(heartbeat, indent=2))


# ─── Queue Management ─────────────────────────────────────────────────────────

def load_queue() -> dict:
    with open(QUEUE_FILE) as f:
        return yaml.safe_load(f)


def save_queue(queue_data: dict):
    with open(QUEUE_FILE, "w") as f:
        yaml.dump(queue_data, f, default_flow_style=False, allow_unicode=True)


def find_next_pending(queue_data: dict) -> Optional[dict]:
    """Retorna o próximo experimento com status: pending, ou None."""
    for exp in queue_data.get("queue", []):
        if exp.get("status") == "pending":
            return exp
    return None


def update_experiment_status(queue_data: dict, exp_id: str,
                              status: str, error: Optional[str] = None,
                              finished_at: Optional[str] = None):
    """Atualiza o status de um experimento no queue_data (in-place)."""
    for exp in queue_data.get("queue", []):
        if exp.get("id") == exp_id:
            exp["status"] = status
            if error:
                exp["error"] = error[:500]  # trunca erros longos
            if finished_at:
                exp["finished_at"] = finished_at
            if status == "running":
                exp["started_at"] = now_iso()
            break


def is_already_done(exp: dict) -> bool:
    """Checa se o aggregate JSON já existe em output_dir (idempotência)."""
    output_dir = exp.get("output_dir")
    if not output_dir:
        return False

    output_path = REPO_ROOT / output_dir

    # Verifica no repo local
    if output_path.exists():
        jsons = list(output_path.glob("aggregate_*.json"))
        if jsons:
            log(f"  [SKIP] Output já existe em {output_path} ({len(jsons)} JSONs)")
            return True

    # Verifica no Drive
    drive_path = Path("/content/drive/MyDrive/seriguela_results") / output_dir
    if drive_path.exists():
        jsons = list(drive_path.glob("aggregate_*.json"))
        if jsons:
            log(f"  [SKIP] Output já existe no Drive em {drive_path} ({len(jsons)} JSONs)")
            return True

    return False


# ─── Execução de Experimento ──────────────────────────────────────────────────

def build_command(exp: dict, defaults: dict) -> list:
    """Constrói o comando de execução a partir do experimento e defaults."""
    script = exp.get("script", defaults.get("script", ""))

    if not script:
        raise ValueError(f"Experimento {exp['id']} não tem 'script' definido")

    cmd = [sys.executable, str(REPO_ROOT / script)]

    args = exp.get("args", {})
    for key, value in args.items():
        if isinstance(value, list):
            # Use nargs="+" style: --flag val1 val2 val3 (not repeated --flag)
            # This is required for argparse nargs="+" arguments like --seeds
            cmd.append(f"--{key}")
            cmd.extend([str(v) for v in value])
        elif isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Adiciona output_dir se definido
    output_dir = exp.get("output_dir")
    if output_dir and "--output_dir" not in cmd:
        cmd.extend(["--output_dir", str(REPO_ROOT / output_dir)])

    return cmd


def run_experiment(exp: dict, defaults: dict) -> tuple[bool, Optional[str]]:
    """
    Executa um experimento.
    Retorna (success: bool, error_message: Optional[str]).
    """
    exp_id = exp.get("id", "unknown")
    log(f"\n{'='*60}")
    log(f"EXECUTANDO: {exp_id}")
    log(f"{'='*60}")

    # Cria output_dir se não existir
    output_dir = exp.get("output_dir")
    if output_dir:
        (REPO_ROOT / output_dir).mkdir(parents=True, exist_ok=True)

    # Constrói e loga o comando
    try:
        cmd = build_command(exp, defaults)
        log(f"Comando: {' '.join(cmd)}")
    except Exception as e:
        return False, str(e)

    # Calcula timeout
    estimated_minutes = exp.get("estimated_minutes", 60)
    max_hours = exp.get("max_hours", defaults.get("max_hours", 2.0))
    timeout_sec = max(estimated_minutes * 60 * 2, max_hours * 3600)

    log(f"Timeout: {timeout_sec/3600:.1f}h")

    # Executa com retry em OOM
    retry_on_oom = exp.get("retry_on_oom", defaults.get("retry_on_oom", True))
    max_attempts = 2 if retry_on_oom else 1

    for attempt in range(1, max_attempts + 1):
        try:
            log(f"Tentativa {attempt}/{max_attempts}...")

            result = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                timeout=timeout_sec,
                capture_output=True,
                text=True
            )

            # Repassa stdout/stderr para o notebook (visibilidade + logging)
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", flush=True)

            if result.returncode == 0:
                log(f"Experimento {exp_id} CONCLUÍDO com sucesso")
                return True, None
            else:
                stderr_tail = result.stderr[-2000:] if result.stderr else ""
                error_msg = f"Exit code {result.returncode}" + (f"\n{stderr_tail}" if stderr_tail else "")
                log(f"Experimento {exp_id} FALHOU: Exit code {result.returncode}")
                if stderr_tail:
                    log(f"Stderr (últimas linhas):\n{stderr_tail}")

                # Checa se foi OOM para tentar com batch menor
                if attempt < max_attempts and "CUDA out of memory" in (result.stderr or ""):
                    log("OOM detectado — reduzindo batch_size pela metade e tentando novamente...")
                    # Modifica o batch_size nos args do experimento
                    if "args" in exp and "batch_size" in exp["args"]:
                        exp["args"]["batch_size"] = exp["args"]["batch_size"] // 2
                        log(f"Novo batch_size: {exp['args']['batch_size']}")
                    continue

                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = f"Timeout após {timeout_sec/3600:.1f}h"
            log(f"Experimento {exp_id} TIMEOUT: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Exceção: {traceback.format_exc()[:500]}"
            log(f"Experimento {exp_id} EXCEÇÃO: {error_msg}")
            return False, error_msg

    return False, "Todas as tentativas falharam"


# ─── Loop Principal ────────────────────────────────────────────────────────────

def run_queue_loop(max_hours: float = 11.0, dry_run: bool = False):
    """
    Loop principal do daemon.
    Processa experimentos pendentes até max_hours ser atingido ou fila acabar.

    Args:
        max_hours: Tempo máximo de execução (default 11h para Colab)
        dry_run: Se True, apenas imprime o que faria sem executar
    """

    # Verifica lock (previne múltiplas instâncias)
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < 3600:  # lock com menos de 1h é válido
            log(f"⚠️  Lock file encontrado ({age/60:.0f} min atrás). Outra instância em execução?")
            log("Se não houver outra instância, delete experiments/_running.lock")
            return

    # Cria lock
    LOCK_FILE.write_text(now_iso())

    start_time = time.time()
    deadline = start_time + (max_hours * 3600)
    last_heartbeat = 0
    last_git_pull = 0
    done_count = 0
    done_since_last_commit = 0

    log(f"{'='*60}")
    log(f"SERIGUELA QUEUE PROCESSOR")
    log(f"Max hours: {max_hours}h | Deadline: {datetime.fromtimestamp(deadline).strftime('%H:%M:%S')}")
    log(f"Dry run: {dry_run}")
    log(f"{'='*60}")

    try:
        while time.time() < deadline:
            current_time = time.time()

            # Git pull periódico para pegar novos jobs
            if current_time - last_git_pull > GIT_PULL_INTERVAL_SEC:
                git_pull()
                last_git_pull = current_time

            # Heartbeat periódico
            if current_time - last_heartbeat > HEARTBEAT_INTERVAL_SEC:
                write_heartbeat(done_count=done_count)
                last_heartbeat = current_time

            # Carrega queue (pode ter mudado após git pull)
            queue_data = load_queue()
            defaults = queue_data.get("defaults", {})

            # Encontra próximo pending
            exp = find_next_pending(queue_data)

            if exp is None:
                log("Fila vazia (nenhum experimento pendente). Aguardando 60s...")
                write_heartbeat(current_exp_id=None, done_count=done_count)
                time.sleep(60)
                continue

            exp_id = exp.get("id", f"exp_{done_count}")
            log(f"\nPróximo experimento: {exp_id}")

            # Checa tempo restante
            time_left = deadline - time.time()
            est_minutes = exp.get("estimated_minutes", 60)

            if time_left < est_minutes * 60 * 1.2:  # 20% de margem
                log(f"⏱️  Tempo insuficiente ({time_left/60:.0f} min restantes, estimado {est_minutes} min). Parando.")
                break

            # Idempotência: já tem resultados?
            if is_already_done(exp):
                update_experiment_status(queue_data, exp_id, "skipped",
                                          finished_at=now_iso())
                save_queue(queue_data)
                log(f"Experimento {exp_id} marcado como skipped (já tem resultados)")
                done_since_last_commit += 1
            elif dry_run:
                log(f"[DRY RUN] Executaria: {exp_id}")
                cmd = build_command(exp, defaults)
                log(f"[DRY RUN] Comando: {' '.join(cmd)}")
                update_experiment_status(queue_data, exp_id, "skipped",
                                          finished_at=now_iso())
                save_queue(queue_data)
            else:
                # Marca como running
                update_experiment_status(queue_data, exp_id, "running")
                save_queue(queue_data)
                write_heartbeat(current_exp_id=exp_id, done_count=done_count)

                # Executa
                success, error = run_experiment(exp, defaults)

                # Atualiza status
                if success:
                    update_experiment_status(queue_data, exp_id, "done", finished_at=now_iso())
                    done_count += 1
                    done_since_last_commit += 1
                    log(f"✅ {exp_id} → done ({done_count} total esta sessão)")
                else:
                    update_experiment_status(queue_data, exp_id, "failed",
                                              error=error, finished_at=now_iso())
                    log(f"❌ {exp_id} → failed: {error}")

                save_queue(queue_data)

            # Commit batch a cada N experimentos
            if done_since_last_commit >= BATCH_COMMIT_EVERY:
                git_commit_batch(f"[queue] {done_since_last_commit} experiments done (total: {done_count})")
                done_since_last_commit = 0

        # Commit final
        if done_since_last_commit > 0:
            git_commit_batch(f"[queue] session end — {done_count} experiments done")

        elapsed = (time.time() - start_time) / 3600
        log(f"\n{'='*60}")
        log(f"SESSÃO ENCERRADA")
        log(f"Tempo: {elapsed:.2f}h | Experimentos done: {done_count}")
        log(f"{'='*60}")

    except KeyboardInterrupt:
        log("\nInterrompido pelo usuário")
        if done_since_last_commit > 0:
            git_commit_batch(f"[queue] interrupted — {done_count} experiments done")

    finally:
        # Remove lock
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
        write_heartbeat(current_exp_id=None, done_count=done_count)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seriguela Queue Processor")
    parser.add_argument("--max-hours", type=float, default=11.0,
                        help="Tempo máximo de execução em horas (default: 11)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas imprime o que faria sem executar")
    parser.add_argument("--status", action="store_true",
                        help="Mostra o status atual da fila e sai")
    args = parser.parse_args()

    if args.status:
        queue_data = load_queue()
        queue = queue_data.get("queue", [])

        from collections import Counter
        status_counts = Counter(e.get("status", "unknown") for e in queue)

        print(f"\nFila: {QUEUE_FILE}")
        print(f"Total: {len(queue)} experimentos")
        print(f"Status: {dict(status_counts)}")
        print()

        for exp in queue:
            status = exp.get("status", "unknown")
            icon = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌", "skipped": "⏭️"}.get(status, "❓")
            print(f"  {icon} {exp.get('id', '?')} [{status}] ~ {exp.get('estimated_minutes', '?')} min")

        print()
        sys.exit(0)

    run_queue_loop(max_hours=args.max_hours, dry_run=args.dry_run)
