"""Orquestrador Google Colab CLI para campanhas v3 (alternativa gratuita/pré-paga ao RunPod).

    python -m seriguela.colab usage
    python -m seriguela.colab run campaigns/g1_timing.json --gpu L4 --budget_cu 10 --parallel 2
    python -m seriguela.colab down

Garantias contra desperdício de Compute Units (CUs):
1. Recusa árvore git suja e commit ausente no GitHub (salvo ``--allow_dirty`` em testes).
2. Confere o saldo de CUs antes de abrir sessão e mede a taxa real ``CU/h`` assim que a VM sobe,
   convertendo ``--budget_cu`` num teto rígido de horas (``max_hours``).
3. Na VM remota: roda G0 (``pytest``) e G1 (``seriguela.canary``) antes da campanha; qualquer
   falha interrompe imediatamente.
4. Persiste resultados em ``/content/results_v3.tgz`` (e opcionalmente espelha cada run no
   Google Drive com ``--drive``) e baixa para a máquina local antes de encerrar.
5. ``try ... finally`` garante ``colab stop -s <session>`` em qualquer saída (sucesso, erro,
   tripwire, Ctrl+C) e verifica que a sessão não ficou ativa.
"""
from __future__ import annotations

import argparse
import io
import json
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import campaign as camp
from .pod import GIT_URL, REPO, check_campaign_committed, check_commit

SESSION_DEFAULT = "seriguela-v3"
SETUP_HOURS = 0.25   # clone + deps + G0 + G1 + download, descontado do orçamento de CUs


class ColabError(RuntimeError):
    """Erro operacional ou de segurança de orçamento no Colab CLI."""


def parse_usage(output: str) -> Dict[str, float]:
    """Extrai saldo, taxa horária e sessões ativas da saída de ``colab usage``."""
    bal = re.search(r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)", output)
    rate = re.search(r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)/hr", output)
    act = re.search(r"Active assignments:\s*([0-9]+)", output)
    if not bal or not rate:
        raise ColabError(f"não consegui interpretar 'colab usage': {output.strip()[:200]}")
    return {
        "balance_cu": float(bal.group(1)),
        "rate_cu_per_hr": float(rate.group(1)),
        "active_assignments": int(act.group(1)) if act else 0,
    }


def budget_cu_hours(budget_cu: float, rate_cu_per_hr: float, setup_hours: float = SETUP_HOURS) -> float:
    """Converte teto de CUs em horas máximas de campanha, descontando o setup."""
    if budget_cu <= 0:
        raise ColabError("budget_cu deve ser positivo")
    if rate_cu_per_hr <= 0:
        # Sessão CPU gratuita ou taxa ainda zero: limita a 2h por segurança padrão
        return round(max(2.0 - setup_hours, 0.25), 3)
    hours = budget_cu / rate_cu_per_hr - setup_hours
    if hours <= 0.05:
        raise ColabError(
            f"orçamento de {budget_cu:.2f} CU não cobre nem o setup a {rate_cu_per_hr:.2f} CU/h"
        )
    return round(hours, 3)


def remote_driver_source(commit: str, campaign_rel: str, out_rel: str, parallel: int,
                         max_hours: float, drive_mirror: Optional[str] = None,
                         skip_canary: bool = False) -> str:
    """Script Python autônomo executado dentro do kernel remoto via ``colab exec -f``."""
    hard_seconds = int((max_hours + SETUP_HOURS) * 3600)
    return f'''import os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

COMMIT = {commit!r}
GIT_URL = {GIT_URL!r}
CAMPAIGN = {campaign_rel!r}
OUT_REL = {out_rel!r}
PARALLEL = {int(parallel)}
MAX_HOURS = {float(max_hours)}
HARD_SECONDS = {hard_seconds}
DRIVE_MIRROR = {drive_mirror!r}
SKIP_CANARY = {bool(skip_canary)}

WORK = Path("/content/seriguela_work")
REPO_DIR = WORK / "seriguela"
ARCHIVE = Path("/content/results_v3.tgz")
SUMMARY = Path("/content/job_summary.json")


def log(msg: str) -> None:
    print(f"[colab {{time.strftime('%H:%M:%S')}}] {{msg}}", flush=True)


def run_checked(cmd, cwd=None, timeout=900):
    log("exec: " + " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        log("STDOUT:\\n" + p.stdout[-1500:])
        log("STDERR:\\n" + p.stderr[-1500:])
        raise RuntimeError(f"comando falhou ({{p.returncode}}): {{cmd[0]}}")
    return p.stdout


def mirror_to_drive():
    if not DRIVE_MIRROR:
        return
    src = REPO_DIR / OUT_REL
    if not src.exists():
        return
    dst = Path(DRIVE_MIRROR) / OUT_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def pack_results():
    src = REPO_DIR / OUT_REL
    if not src.exists():
        return False
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(src, arcname=OUT_REL)
    mirror_to_drive()
    return True


started = time.time()
outcome = "unknown"
try:
    WORK.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        run_checked(["git", "clone", "-q", GIT_URL, str(REPO_DIR)])
    run_checked(["git", "fetch", "-q", "origin"], cwd=REPO_DIR)
    run_checked(["git", "checkout", "-q", "--detach", COMMIT], cwd=REPO_DIR)
    head = run_checked(["git", "rev-parse", "HEAD"], cwd=REPO_DIR).strip()
    assert head == COMMIT, f"HEAD {{head}} != {{COMMIT}}"

    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    run_checked([sys.executable, "-m", "pip", "install", "-q",
                 "transformers>=4.51", "peft>=0.15", "sympy>=1.13.1",
                 "scipy>=1.11", "pandas>=2.2", "pytest>=8"],
                cwd=REPO_DIR, timeout=900)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = str(Path(DRIVE_MIRROR) / "_hf_cache" if DRIVE_MIRROR else WORK / "hf_cache")
    gpu_out = run_checked([sys.executable, "-c",
                           "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
                          cwd=REPO_DIR, timeout=120).strip()
    log(f"GPU ativa: {{gpu_out}}")

    log("Portão G0: pytest")
    run_checked([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:warnings"],
                cwd=REPO_DIR, timeout=600)

    if not SKIP_CANARY:
        log("Portão G1: canário")
        run_checked([sys.executable, "-m", "seriguela.canary", "--steps", "4", "--batch", "128"],
                    cwd=REPO_DIR, timeout=900)

    log(f"Iniciando campanha {{CAMPAIGN}} (parallel={{PARALLEL}}, max_hours={{MAX_HOURS}})")
    proc = subprocess.Popen(
        [sys.executable, "-m", "seriguela.campaign", "work", CAMPAIGN,
         "--parallel", str(PARALLEL), "--max_hours", str(MAX_HOURS)],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    last_beat = 0.0
    while proc.poll() is None:
        if time.time() - started > HARD_SECONDS:
            proc.kill()
            proc.wait()
            outcome = "hard_timeout"
            break
        if time.time() - last_beat >= 15.0:
            st_file = REPO_DIR / OUT_REL / "campaign_status.json"
            if st_file.exists():
                log("status: " + st_file.read_text(encoding="utf-8").replace("\\n", " "))
            else:
                log("aguardando primeiro snapshot da campanha...")
            mirror_to_drive()
            last_beat = time.time()
        time.sleep(3.0)
    else:
        tail = (proc.stdout.read() if proc.stdout else "")[-1000:]
        if tail.strip():
            log(tail.strip())
        outcome = "complete" if proc.returncode == 0 else f"campaign_rc_{{proc.returncode}}"
except Exception as exc:
    outcome = f"error: {{exc}}"
    log(outcome)
finally:
    packed = pack_results()
    import json as _json
    SUMMARY.write_text(_json.dumps({{"outcome": outcome, "packed": packed,
                                     "elapsed_s": round(time.time() - started, 1)}}), encoding="utf-8")
    log(f"FIM: outcome={{outcome}} packed={{packed}}")
'''


RunnerFn = Callable[[Sequence[str], int], subprocess.CompletedProcess]


def _default_runner(cmd: Sequence[str], timeout: int = 3600, idle_timeout: int = 240) -> subprocess.CompletedProcess:
    exe = shutil.which("colab")
    if not exe:
        raise ColabError("'colab' não encontrado no PATH (instale com: uv tool install google-colab-cli)")
    if cmd and cmd[0] == "exec":
        # Transmite o progresso ao vivo com watchdog de inatividade (o colab-cli pode travar
        # sem fechar stdout quando o websocket cai com 'Connection was lost.').
        proc = subprocess.Popen([exe, *cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        q: "queue.Queue[Optional[str]]" = queue.Queue()

        def _reader():
            assert proc.stdout is not None
            for ln in proc.stdout:
                q.put(ln)
            q.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        lines: List[str] = []
        started = last_io = time.time()
        while True:
            now = time.time()
            if now - started > timeout:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired([exe, *cmd], timeout)
            if now - last_io > idle_timeout:
                proc.kill()
                proc.wait()
                raise ColabError(f"colab exec sem saída por {idle_timeout}s (possível travamento de websocket)")
            try:
                line = q.get(timeout=2.0)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            last_io = time.time()
            print(line, end="", flush=True)
            lines.append(line)
            if "Connection was lost." in line:
                proc.kill()
                proc.wait()
                raise ColabError("colab exec perdeu conexão websocket ('Connection was lost.')")
        proc.wait()
        return subprocess.CompletedProcess([exe, *cmd], proc.returncode, stdout="".join(lines), stderr="")
    return subprocess.run([exe, *cmd], text=True, capture_output=True, timeout=timeout)


def usage(runner: RunnerFn = _default_runner) -> Dict[str, float]:
    res = runner(["usage"], 60)
    if res.returncode != 0:
        raise ColabError(f"colab usage falhou: {res.stderr.strip() or res.stdout.strip()}")
    return parse_usage(res.stdout)


def stop_session(session: str = SESSION_DEFAULT, runner: RunnerFn = _default_runner) -> None:
    runner(["stop", "-s", session], 120)


def run_campaign(campaign_path: str, gpu: str = "L4", budget_cu: float = 10.0,
                 parallel: int = 2, session: str = SESSION_DEFAULT, drive: bool = False,
                 high_mem: bool = False, skip_canary: bool = False, allow_dirty: bool = False,
                 runner: RunnerFn = _default_runner) -> Dict:
    """Executa uma campanha no Colab com teto de CUs e desligamento garantido."""
    spec = camp.load(campaign_path)
    camp.expand(spec)
    if allow_dirty:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
    else:
        commit = check_commit()
        check_campaign_committed(commit, campaign_path)

    before = usage(runner)
    if before["balance_cu"] < budget_cu:
        raise ColabError(f"saldo insuficiente: {before['balance_cu']:.2f} CU < budget_cu {budget_cu:.2f} CU")

    new_args = ["new", "-s", session, "--gpu", gpu] + (["--high-mem"] if high_mem else [])
    created = runner(new_args, 300)
    if created.returncode != 0:
        raise ColabError(f"falha ao criar sessão Colab ({gpu}): {created.stderr.strip() or created.stdout.strip()}")

    summary: Dict = {"session": session, "gpu": gpu, "commit": commit, "campaign": spec["name"]}
    try:
        active = usage(runner)
        rate = max(active["rate_cu_per_hr"] - before["rate_cu_per_hr"], active["rate_cu_per_hr"])
        max_hours = budget_cu_hours(budget_cu, rate)
        summary.update(rate_cu_per_hr=rate, max_hours=max_hours, balance_start_cu=before["balance_cu"])

        drive_mirror = None
        if drive:
            dm = runner(["drivemount", "-s", session], 180)
            if dm.returncode != 0:
                raise ColabError(f"falha ao montar Google Drive: {dm.stderr.strip()}")
            drive_mirror = "/content/drive/MyDrive/seriguela_v3"

        camp_rel = Path(campaign_path).resolve().relative_to(REPO).as_posix()
        driver_code = remote_driver_source(
            commit=commit, campaign_rel=camp_rel, out_rel=spec["out"],
            parallel=parallel, max_hours=max_hours, drive_mirror=drive_mirror,
            skip_canary=skip_canary,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Espera o kernel recém-alocado estabilizar (evita 'Connection was lost' nos primeiros segundos)
            probe_file = tmp_dir / "probe.py"
            probe_file.write_text("print('KERNEL_READY', flush=True)\n", encoding="utf-8")
            for attempt in range(3):
                try:
                    pr = runner(["exec", "-s", session, "--timeout", "45", "-f", str(probe_file)], 90)
                    if pr.returncode == 0:
                        break
                except ColabError:
                    if attempt == 2:
                        raise
                time.sleep(5)
            driver_file = tmp_dir / "colab_driver.py"
            driver_file.write_text(driver_code, encoding="utf-8")
            hard_wall = int((max_hours + SETUP_HOURS + 0.25) * 3600)
            exec_res = runner(["exec", "-s", session, "--timeout", "0", "-f", str(driver_file)], hard_wall)
            summary["exec_rc"] = exec_res.returncode

            sum_local = tmp_dir / "job_summary.json"
            if runner(["download", "-s", session, "/content/job_summary.json", str(sum_local)], 120).returncode == 0 \
                    and sum_local.exists():
                summary.update(json.loads(sum_local.read_text(encoding="utf-8")))

            if summary.get("packed"):
                REPO.joinpath("outputs").mkdir(parents=True, exist_ok=True)
                tgz_local = REPO / "outputs" / f"{spec['name']}_{int(time.time())}.tgz"
                dl = runner(["download", "-s", session, "/content/results_v3.tgz", str(tgz_local)], 900)
                if dl.returncode != 0 or not tgz_local.exists():
                    raise ColabError(f"falha ao baixar /content/results_v3.tgz: {dl.stderr.strip()}")
                with tarfile.open(tgz_local, mode="r:gz") as tar:
                    tar.extractall(REPO, filter="data")
                summary["archive"] = str(tgz_local)
    finally:
        stop_session(session, runner)
        try:
            after = usage(runner)
            summary["balance_end_cu"] = after["balance_cu"]
            summary["spent_cu"] = round(before["balance_cu"] - after["balance_cu"], 3)
            summary["active_after_stop"] = after["active_assignments"]
        except Exception:
            pass
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("usage")

    p_down = sub.add_parser("down")
    p_down.add_argument("-s", "--session", default=SESSION_DEFAULT)

    p_run = sub.add_parser("run")
    p_run.add_argument("campaign")
    p_run.add_argument("--gpu", default="L4", choices=["T4", "L4", "G4", "A100", "H100"])
    p_run.add_argument("--budget_cu", type=float, required=True, help="teto máximo em Compute Units")
    p_run.add_argument("--parallel", type=int, default=2)
    p_run.add_argument("-s", "--session", default=SESSION_DEFAULT)
    p_run.add_argument("--drive", action="store_true", help="espelha resultados no Google Drive")
    p_run.add_argument("--high_mem", action="store_true")
    p_run.add_argument("--skip_canary", action="store_true")
    p_run.add_argument("--allow_dirty", action="store_true", help="apenas para testes locais")

    args = parser.parse_args(argv)
    try:
        if args.action == "usage":
            print(json.dumps(usage(), indent=2))
        elif args.action == "down":
            stop_session(args.session)
            print(json.dumps(usage(), indent=2))
        elif args.action == "run":
            out = run_campaign(
                args.campaign, gpu=args.gpu, budget_cu=args.budget_cu, parallel=args.parallel,
                session=args.session, drive=args.drive, high_mem=args.high_mem,
                skip_canary=args.skip_canary, allow_dirty=args.allow_dirty,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0 if out.get("outcome") == "complete" else 1
    except ColabError as err:
        print(f"ERRO Colab: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
