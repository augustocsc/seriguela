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
4. Execução desacoplada (``start_new_session=True``): a campanha roda em background na VM
   independente do websocket do Jupyter. O cliente local faz *polling* curto a cada 45s (imune a
   quedas de websocket após ~2h), baixa incrementalmente cada run concluído para o disco local e
   envia runs locais já concluídos (``colab upload``) ao retomar.
5. ``try ... finally`` tenta um download de resgate antes de garantir ``colab stop -s <session>``
   em qualquer saída (sucesso, erro, tripwire, Ctrl+C) e verifica que a sessão não ficou ativa.
"""
from __future__ import annotations

import argparse
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
POLL_PREFIX = "POLL_JSON:"


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
    """Setup + G0 + G1 + disparo desacoplado de ``seriguela.campaign work`` na VM Colab."""
    hard_seconds = int((max_hours + SETUP_HOURS) * 3600)
    bg_script = f'''import json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

REPO_DIR = Path("/content/seriguela_work/seriguela")
CAMPAIGN = {campaign_rel!r}
OUT_REL = {out_rel!r}
PARALLEL = {int(parallel)}
MAX_HOURS = {float(max_hours)}
HARD_SECONDS = {hard_seconds}
DRIVE_MIRROR = {drive_mirror!r}
ARCHIVE = Path("/content/results_v3.tgz")
PACKED_DONE = Path("/content/packed_done.txt")
SUMMARY = Path("/content/job_summary.json")


def mirror_to_drive():
    if not DRIVE_MIRROR:
        return
    src = REPO_DIR / OUT_REL
    if src.exists():
        dst = Path(DRIVE_MIRROR) / OUT_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def pack_results(done_count: int = -1) -> bool:
    src = REPO_DIR / OUT_REL
    if not src.exists():
        return False
    tmp = ARCHIVE.with_suffix(".tgz.tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(src, arcname=OUT_REL)
    os.replace(tmp, ARCHIVE)
    if done_count >= 0:
        PACKED_DONE.write_text(str(done_count), encoding="utf-8")
    mirror_to_drive()
    return True


started = time.time()
outcome = "unknown"
last_packed = -1
try:
    proc = subprocess.Popen(
        [sys.executable, "-m", "seriguela.campaign", "work", CAMPAIGN,
         "--parallel", str(PARALLEL), "--max_hours", str(MAX_HOURS)],
        cwd=REPO_DIR,
    )
    while proc.poll() is None:
        if time.time() - started > HARD_SECONDS:
            proc.kill()
            proc.wait()
            outcome = "hard_timeout"
            break
        st_file = REPO_DIR / OUT_REL / "campaign_status.json"
        if st_file.exists():
            try:
                st = json.loads(st_file.read_text(encoding="utf-8"))
                done_now = int(st.get("done", 0))
                if done_now > last_packed:
                    if pack_results(done_now):
                        last_packed = done_now
            except Exception:
                pass
        time.sleep(10.0)
    else:
        outcome = "complete" if proc.returncode == 0 else f"campaign_rc_{{proc.returncode}}"
except Exception as exc:
    outcome = f"error: {{exc}}"
finally:
    done_final = last_packed
    st_file = REPO_DIR / OUT_REL / "campaign_status.json"
    if st_file.exists():
        try:
            done_final = int(json.loads(st_file.read_text(encoding="utf-8")).get("done", last_packed))
        except Exception:
            pass
    packed = pack_results(done_final)
    SUMMARY.write_text(json.dumps({{"outcome": outcome, "packed": packed,
                                     "done": done_final,
                                     "elapsed_s": round(time.time() - started, 1)}}),
                       encoding="utf-8")
'''
    return f'''import os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

COMMIT = {commit!r}
GIT_URL = {GIT_URL!r}
CAMPAIGN = {campaign_rel!r}
OUT_REL = {out_rel!r}
PARALLEL = {int(parallel)}
MAX_HOURS = {float(max_hours)}
DRIVE_MIRROR = {drive_mirror!r}
SKIP_CANARY = {bool(skip_canary)}

WORK = Path("/content/seriguela_work")
REPO_DIR = WORK / "seriguela"
SEED_TGZ = Path("/content/seed_results.tgz")
BG_SCRIPT = Path("/content/run_bg.py")
BG_LOG = Path("/content/campaign.log")
BG_PID = Path("/content/campaign.pid")
SUMMARY = Path("/content/job_summary.json")
PACKED_DONE = Path("/content/packed_done.txt")


def log(msg: str) -> None:
    print(f"[colab {{time.strftime('%H:%M:%S')}}] {{msg}}", flush=True)


def run_checked(cmd, cwd=None, timeout=900, stream=False):
    log("exec: " + " ".join(str(x) for x in cmd))
    if stream:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None
        for ln in p.stdout:
            print(ln, end="", flush=True)
        p.wait(timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"comando falhou ({{p.returncode}}): {{cmd[0]}}")
        return ""
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        log("STDOUT:\\n" + p.stdout[-1500:])
        log("STDERR:\\n" + p.stderr[-1500:])
        raise RuntimeError(f"comando falhou ({{p.returncode}}): {{cmd[0]}}")
    return p.stdout


WORK.mkdir(parents=True, exist_ok=True)
for stale in (SUMMARY, PACKED_DONE, BG_PID):
    if stale.exists():
        stale.unlink()

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
            cwd=REPO_DIR, timeout=600, stream=True)

if not SKIP_CANARY:
    log("Portão G1: canário")
    run_checked([sys.executable, "-m", "seriguela.canary", "--steps", "4", "--batch", "128"],
                cwd=REPO_DIR, timeout=900, stream=True)

if SEED_TGZ.exists():
    with tarfile.open(SEED_TGZ, "r:gz") as tar:
        tar.extractall(REPO_DIR)
    log("retomando runs locais enviados via seed_results.tgz")

if DRIVE_MIRROR:
    prev = Path(DRIVE_MIRROR) / OUT_REL
    if prev.exists():
        dst_local = REPO_DIR / OUT_REL
        dst_local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prev, dst_local, dirs_exist_ok=True)
        log(f"retomando runs salvos no Drive: {{prev}}")

BG_SCRIPT.write_text({bg_script!r}, encoding="utf-8")
log_fh = open(BG_LOG, "w", encoding="utf-8")
bg = subprocess.Popen(
    [sys.executable, str(BG_SCRIPT)],
    cwd=REPO_DIR, stdout=log_fh, stderr=subprocess.STDOUT,
    start_new_session=True,
)
BG_PID.write_text(str(bg.pid), encoding="utf-8")
log(f"Campanha {{CAMPAIGN}} iniciada em background (pid={{bg.pid}}, parallel={{PARALLEL}}, max_hours={{MAX_HOURS}})")
'''


def remote_poll_source(out_rel: str) -> str:
    """Consulta leve (~1s) do estado da campanha em background na VM Colab."""
    return f'''import json, os
from pathlib import Path

pid_file = Path("/content/campaign.pid")
sum_file = Path("/content/job_summary.json")
st_file = Path("/content/seriguela_work/seriguela") / {out_rel!r} / "campaign_status.json"
pk_file = Path("/content/packed_done.txt")

alive = False
if pid_file.exists():
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        alive = True
    except Exception:
        alive = False

payload = {{
    "alive": alive,
    "summary": json.loads(sum_file.read_text(encoding="utf-8")) if sum_file.exists() else None,
    "status": json.loads(st_file.read_text(encoding="utf-8")) if st_file.exists() else None,
    "packed_done": int(pk_file.read_text(encoding="utf-8").strip()) if pk_file.exists() else -1,
}}
print({POLL_PREFIX!r} + json.dumps(payload), flush=True)
'''


def parse_poll_output(stdout: str) -> Dict:
    for line in reversed(stdout.splitlines()):
        if POLL_PREFIX in line:
            raw = line.split(POLL_PREFIX, 1)[1].strip()
            return json.loads(raw)
    raise ColabError(f"saída de poll sem {POLL_PREFIX}: {stdout.strip()[-200:]}")


RunnerFn = Callable[[Sequence[str], int], subprocess.CompletedProcess]


def _default_runner(cmd: Sequence[str], timeout: int = 3600, idle_timeout: int = 240) -> subprocess.CompletedProcess:
    exe = shutil.which("colab")
    if not exe:
        raise ColabError("'colab' não encontrado no PATH (instale com: uv tool install google-colab-cli)")
    if cmd and cmd[0] == "exec":
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


def _download_and_extract(session: str, name: str, runner: RunnerFn) -> Optional[str]:
    """Baixa ``/content/results_v3.tgz`` e extrai em ``REPO``. Retorna caminho do arquivo ou None."""
    REPO.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    tgz_local = REPO / "outputs" / f"{name}_{int(time.time())}.tgz"
    dl = runner(["download", "-s", session, "/content/results_v3.tgz", str(tgz_local)], 900)
    if dl.returncode != 0 or not tgz_local.exists():
        return None
    with tarfile.open(tgz_local, mode="r:gz") as tar:
        tar.extractall(REPO, filter="data")
    return str(tgz_local)


def _upload_seed_results(session: str, out_rel: str, tmp_dir: Path, runner: RunnerFn) -> int:
    """Se já houver runs concluídos localmente em ``out_rel``, envia para a VM retomar."""
    local_out = REPO / out_rel
    if not local_out.exists():
        return 0
    existing = list(local_out.rglob("seed_*.json"))
    if not existing:
        return 0
    seed_tgz = tmp_dir / "seed_results.tgz"
    with tarfile.open(seed_tgz, "w:gz") as tar:
        tar.add(local_out, arcname=out_rel)
    up = runner(["upload", "-s", session, str(seed_tgz), "/content/seed_results.tgz"], 300)
    if up.returncode != 0:
        raise ColabError(f"falha ao enviar runs locais para retomada: {up.stderr.strip() or up.stdout.strip()}")
    return len(existing)


def run_campaign(campaign_path: str, gpu: str = "L4", budget_cu: float = 10.0,
                 parallel: int = 2, session: str = SESSION_DEFAULT, drive: bool = False,
                 high_mem: bool = False, skip_canary: bool = False, allow_dirty: bool = False,
                 poll_interval: float = 45.0,
                 runner: RunnerFn = _default_runner) -> Dict:
    """Executa uma campanha no Colab com teto de CUs, sync incremental e desligamento garantido."""
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
    final_synced = False
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
        poll_code = remote_poll_source(spec["out"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            resumed = _upload_seed_results(session, spec["out"], tmp_dir, runner)
            if resumed:
                summary["resumed_local_runs"] = resumed
                print(f"[local] enviados {resumed} runs concluídos para retomada na VM", flush=True)

            # Espera o kernel recém-alocado estabilizar
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
                time.sleep(min(5.0, poll_interval))

            # Fase 1: setup + G0 + G1 + launch em background (~4-6 min)
            driver_file = tmp_dir / "colab_driver.py"
            driver_file.write_text(driver_code, encoding="utf-8")
            setup_res = runner(["exec", "-s", session, "--timeout", "1800", "-f", str(driver_file)], 2100)
            summary["setup_rc"] = setup_res.returncode
            if setup_res.returncode != 0:
                raise ColabError(f"setup/G0/G1 falhou no Colab (rc={setup_res.returncode})")

            # Fase 2: polling curto + download incremental a cada run concluído
            poll_file = tmp_dir / "poll.py"
            poll_file.write_text(poll_code, encoding="utf-8")
            hard_deadline = time.time() + (max_hours + SETUP_HOURS + 0.25) * 3600
            last_synced_done = resumed
            consecutive_errors = 0

            while time.time() < hard_deadline:
                if poll_interval > 0:
                    time.sleep(poll_interval)
                try:
                    p_res = runner(["exec", "-s", session, "--timeout", "60", "-f", str(poll_file)], 120)
                    if p_res.returncode != 0:
                        raise ColabError(f"poll rc={p_res.returncode}")
                    state = parse_poll_output(p_res.stdout)
                    consecutive_errors = 0
                except (ColabError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                    consecutive_errors += 1
                    print(f"[local] aviso de poll ({consecutive_errors}/8): {exc}", flush=True)
                    if consecutive_errors >= 8:
                        raise ColabError(f"8 falhas consecutivas de poll: {exc}") from exc
                    if poll_interval > 0:
                        time.sleep(min(15.0, poll_interval))
                    continue

                st = state.get("status")
                if st:
                    print(
                        f"[local {time.strftime('%H:%M:%S')}] done={st.get('done')}/{st.get('total')} "
                        f"running={st.get('running')} eta_h={st.get('eta_h')} "
                        f"failures={len(st.get('failures') or [])}",
                        flush=True,
                    )

                packed_done = int(state.get("packed_done", -1))
                if packed_done > last_synced_done:
                    arc = _download_and_extract(session, spec["name"], runner)
                    if arc:
                        last_synced_done = packed_done
                        summary["archive"] = arc
                        print(f"[local] checkpoint sincronizado (done={packed_done}) -> {arc}", flush=True)

                if state.get("summary") is not None:
                    summary.update(state["summary"])
                    break
                if not state.get("alive"):
                    break

            # Fase 3: download final garantido
            sum_local = tmp_dir / "job_summary.json"
            if runner(["download", "-s", session, "/content/job_summary.json", str(sum_local)], 120).returncode == 0 \
                    and sum_local.exists():
                summary.update(json.loads(sum_local.read_text(encoding="utf-8")))

            arc = _download_and_extract(session, spec["name"], runner)
            if arc:
                summary["archive"] = arc
                final_synced = True
            elif summary.get("packed"):
                raise ColabError("falha ao baixar /content/results_v3.tgz final")
    finally:
        if not final_synced:
            try:
                arc = _download_and_extract(session, spec["name"], runner)
                if arc:
                    summary["archive"] = arc
                    summary["rescued_on_error"] = True
            except Exception:
                pass
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
    p_run.add_argument("--poll_interval", type=float, default=45.0, help="intervalo de polling em segundos")

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
                poll_interval=args.poll_interval,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0 if out.get("outcome") == "complete" else 1
    except ColabError as err:
        print(f"ERRO Colab: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
