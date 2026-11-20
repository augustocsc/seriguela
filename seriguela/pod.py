"""Controle do pod RunPod para uma campanha (só biblioteca padrão; roda na máquina local).

    python -m seriguela.pod up campaigns/g1_timing.json --budget 2 --parallel 3
    python -m seriguela.pod status
    python -m seriguela.pod watch          # acompanha, busca os resultados e desliga
    python -m seriguela.pod fetch
    python -m seriguela.pod down

Garantias contra gasto à toa:
1. ``up`` recusa árvore suja e commit que não está no GitHub: o pod roda exatamente o
   commit local.
2. O teto de gasto vira um teto de horas (``--budget`` / preço por hora). No pod, tudo
   roda sob ``timeout``; ao terminar (bem ou mal), o pod se desliga sozinho.
3. ``up`` confere, antes de qualquer GPU-hora útil, que o pod consegue se desligar
   (``runpodctl`` com a chave do próprio pod). Se não consegue, o pod é apagado na hora.
4. No pod: G0 (pytest) e G1 (canário) antes da campanha. Qualquer falha encerra.
5. Ao fim, o pod espera ``--grace`` minutos que ``fetch`` (ou ``watch``) busque os
   resultados e então se apaga. Se ninguém buscar, ele PARA em vez de apagar: o volume
   em /workspace fica guardado (custo de disco, sem GPU).

A chave da API fica em ``~/.tokens.txt`` (linha ``runpod = ...``) e nunca é impressa nem
enviada ao pod.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"
POD_NAME = "seriguela-v3"
GPUS = ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GIT_URL = "https://github.com/augustocsc/seriguela.git"
INFO = REPO / "outputs" / "pod_info.json"
SSH_KEY = Path.home() / ".ssh" / "runpod_seriguela"
SETUP_HOURS = 0.5            # bootstrap + G0 + G1 + espera da busca, descontado do orçamento
WORK = "/workspace"


class PodError(RuntimeError):
    pass


# --------------------------------------------------------------------- partes puras (testadas)

def budget_hours(budget_usd: float, price_per_hour: float, grace_min: float) -> float:
    """Horas de campanha que cabem no orçamento depois do bootstrap e da espera final."""
    if price_per_hour <= 0:
        raise PodError("preço por hora inválido")
    hours = budget_usd / price_per_hour - SETUP_HOURS - grace_min / 60
    if hours <= 0.05:
        raise PodError(f"orçamento de US$ {budget_usd} não cobre nem o bootstrap a US$ {price_per_hour}/h")
    return round(hours, 3)


def extract_ssh(pod: Dict) -> Tuple[Optional[str], Optional[int]]:
    ip = pod.get("publicIp") or ""
    ports = pod.get("portMappings") or {}
    port = ports.get("22") or ports.get(22)
    return (ip, int(port)) if ip and port else (None, None)


def job_script(commit: str, campaign: str, parallel: int, max_hours: float, grace_min: float,
               work: str = WORK) -> str:
    """Script que roda no pod. A primeira linha útil fixa o commit; o resto é idempotente."""
    hard = int((max_hours + SETUP_HOURS) * 3600)
    return f"""#!/bin/bash
# gerado por seriguela.pod; não edite no pod
set -u
cd {work}
log() {{ echo "[$(date -u +%H:%M:%S)] $*"; }}
self_off() {{  # $1 = remove | stop
  log "desligando o pod: $1"; runpodctl "$1" pod "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID"
}}
finish() {{
  pkill -f "seriguela[.]" 2>/dev/null || true
  echo "$1" > {work}/JOB_DONE; log "fim: $1"
  waited=0
  while [ ! -f {work}/FETCHED ] && [ $waited -lt {int(grace_min * 60)} ]; do sleep 15; waited=$((waited+15)); done
  if [ -f {work}/FETCHED ]; then self_off remove; else self_off stop; fi
  exit 0
}}
inner() {{
  set -e
  if [ ! -d seriguela ]; then git clone -q {GIT_URL} seriguela; fi
  cd seriguela && git fetch -q origin && git checkout -q --detach {commit}
  test "$(git rev-parse HEAD)" = "{commit}"
  if [ ! -f {work}/venv/ok ]; then
    python -m venv {work}/venv && . {work}/venv/bin/activate
    pip install -q --upgrade pip && pip install -q -r requirements.txt
    pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    touch {work}/venv/ok
  fi
  . {work}/venv/bin/activate
  export HF_HOME={work}/hf
  python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
  log "G0: pytest"; python -m pytest -q -x -p no:warnings || {{ echo G0_FALHOU; return 10; }}
  log "G1: canário"; python -m seriguela.canary --steps 4 --batch 128 || {{ echo G1_FALHOU; return 11; }}
  log "campanha {campaign}"
  python -m seriguela.campaign work {campaign} --parallel {parallel} --max_hours {max_hours}
}}
export -f inner log
timeout {hard} bash -c inner > {work}/job.log 2>&1
rc=$?
case $rc in
  0) finish complete;; 1) finish campaign_stopped;; 10) finish g0_failed;; 11) finish g1_failed;;
  124) finish hard_timeout;; *) finish "error_$rc";;
esac
"""


# --------------------------------------------------------------------- ambiente local

def _key() -> str:
    for line in (Path.home() / ".tokens.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().lower().startswith("runpod"):
            return line.split("=", 1)[1].strip()
    raise PodError("linha 'runpod = <chave>' não encontrada em ~/.tokens.txt")


def _api(method: str, path: str, body: Optional[Dict] = None):
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as error:
        print(f"[API {error.code}] {method} {path}: {error.read().decode(errors='ignore')[:500]}", file=sys.stderr)
        return None


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def check_commit() -> str:
    """O commit local, desde que limpo e já no GitHub."""
    if _git("status", "--porcelain", "--untracked-files=no") or _git("status", "--porcelain", "--", "seriguela"):
        raise PodError("árvore git suja: faça commit antes de subir um pod")
    commit = _git("rev-parse", "HEAD")
    _git("fetch", "-q", "origin")
    if not _git("branch", "-r", "--contains", commit):
        raise PodError(f"o commit {commit[:8]} não está no GitHub: faça push antes")
    return commit


def check_campaign_committed(commit: str, campaign: str) -> None:
    rel = Path(campaign).resolve().relative_to(REPO).as_posix()
    if subprocess.run(["git", "cat-file", "-e", f"{commit}:{rel}"], cwd=REPO).returncode != 0:
        raise PodError(f"{rel} não está no commit {commit[:8]}: o pod não o veria")


_EMPTY_CONFIG = REPO / "outputs" / "ssh_config.empty"


def _ssh_args(info: Dict) -> list:
    _EMPTY_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _EMPTY_CONFIG.touch()
    return ["-F", str(_EMPTY_CONFIG), "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
            "-o", "BatchMode=yes", "-i", str(SSH_KEY), "-p", str(info["port"]), f"root@{info['ip']}"]


def ssh(info: Dict, command: str, data: Optional[bytes] = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *_ssh_args(info), command], input=data, capture_output=True, timeout=timeout)


def _info() -> Dict:
    if not INFO.exists():
        raise PodError("nenhum pod registrado (outputs/pod_info.json); use 'up'")
    return json.loads(INFO.read_text(encoding="utf-8"))


def _find_pod() -> Optional[Dict]:
    pods = _api("GET", "/pods") or []
    pods = pods if isinstance(pods, list) else pods.get("pods", [])
    return next((p for p in pods if p.get("name") == POD_NAME), None)


# --------------------------------------------------------------------- comandos

def up(campaign: str, budget: float, parallel: int, grace: float) -> None:
    from . import campaign as camp
    spec = camp.load(campaign)
    n = len(camp.expand(spec))
    commit = check_commit()
    check_campaign_committed(commit, campaign)
    if _find_pod():
        raise PodError(f"já existe um pod '{POD_NAME}': use status/watch/down")
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(SSH_KEY)], check=True)
    pubkey = SSH_KEY.with_suffix(".pub").read_text().strip()
    pod = None
    for gpu in GPUS:
        print(f"tentando {gpu} (community)...", flush=True)
        pod = _api("POST", "/pods", {
            "name": POD_NAME, "imageName": IMAGE, "cloudType": "COMMUNITY", "computeType": "GPU",
            "gpuCount": 1, "gpuTypeIds": [gpu], "containerDiskInGb": 30, "volumeInGb": 20,
            "volumeMountPath": WORK, "ports": ["22/tcp"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": pubkey}})
        if pod and pod.get("id"):
            break
        time.sleep(3)
    if not pod or not pod.get("id"):
        raise PodError("nenhuma GPU disponível na lista")
    price = float(pod.get("costPerHr") or 0)
    info = {"pod_id": pod["id"], "gpu": gpu, "price": price, "campaign": campaign, "out": spec["out"],
            "commit": commit, "created": time.time()}
    INFO.parent.mkdir(parents=True, exist_ok=True)
    INFO.write_text(json.dumps(info, indent=1), encoding="utf-8")
    try:
        hours = budget_hours(budget, price, grace)
        print(f"pod {pod['id']} {gpu} US$ {price}/h -> {hours} h de campanha ({n} runs, {parallel} por vez)")
        for _ in range(60):
            ip, port = extract_ssh(_api("GET", f"/pods/{pod['id']}") or {})
            if ip:
                break
            time.sleep(10)
        else:
            raise PodError("o pod não expôs SSH em 10 minutos")
        info.update(ip=ip, port=port, hours=hours)
        INFO.write_text(json.dumps(info, indent=1), encoding="utf-8")
        for _ in range(30):
            if ssh(info, "echo ok", timeout=30).returncode == 0:
                break
            time.sleep(10)
        check = ssh(info, 'command -v runpodctl >/dev/null && test -n "$RUNPOD_POD_ID" && '
                          'runpodctl get pod "$RUNPOD_POD_ID" >/dev/null && echo SELF_OFF_OK', timeout=60)
        if b"SELF_OFF_OK" not in check.stdout:
            raise PodError("o pod não consegue se desligar sozinho (runpodctl/RUNPOD_POD_ID): abortado")
        script = job_script(commit, campaign, parallel, hours, grace).encode()
        ssh(info, f"cat > {WORK}/job.sh && chmod +x {WORK}/job.sh", data=script, timeout=60)
        started = ssh(info, f"rm -f {WORK}/JOB_DONE {WORK}/FETCHED; "
                            f"setsid nohup {WORK}/job.sh > {WORK}/job_outer.log 2>&1 < /dev/null & echo STARTED")
        if b"STARTED" not in started.stdout:
            raise PodError("não confirmei o início do job no pod")
    except BaseException:
        print("erro na subida: apagando o pod", file=sys.stderr)
        down()
        raise
    print("job iniciado. Acompanhe com: python -m seriguela.pod watch")


def status(verbose: bool = True) -> Dict:
    info = _info()
    pod = _api("GET", f"/pods/{info['pod_id']}") or {}
    state = {"pod": pod.get("desiredStatus", "ausente"),
             "custo_ate_agora_usd": round(info["price"] * (time.time() - info["created"]) / 3600, 3)}
    if state["pod"] == "RUNNING" and "ip" in info:
        out = info["out"]
        r = ssh(info, f"cat {WORK}/JOB_DONE 2>/dev/null; echo '---'; "
                      f"cat {WORK}/seriguela/{out}/campaign_status.json 2>/dev/null; echo '---'; "
                      f"tail -5 {WORK}/job.log 2>/dev/null", timeout=60)
        done, camp, tail = (r.stdout.decode(errors="ignore").split("---\n") + ["", "", ""])[:3]
        state["job_done"] = done.strip() or None
        try:
            state["campanha"] = json.loads(camp)
        except ValueError:
            state["campanha"] = None
        state["log"] = tail.strip()
    if verbose:
        print(json.dumps(state, indent=1, ensure_ascii=False))
    return state


def fetch() -> Path:
    info = _info()
    out = info["out"]
    r = ssh(info, f"cd {WORK}/seriguela && tar czf - {out} && touch {WORK}/FETCHED", timeout=1800)
    if r.returncode != 0:
        raise PodError(f"falha ao buscar resultados: {r.stderr.decode(errors='ignore')[:300]}")
    backup = REPO / "outputs" / f"{Path(out).name}_{int(time.time())}.tgz"
    backup.write_bytes(r.stdout)
    with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:gz") as tar:
        tar.extractall(REPO, filter="data")
    print(f"resultados em {REPO / out} (cópia em {backup})")
    return REPO / out


def down() -> None:
    pod = _find_pod()
    if not pod:
        print("nenhum pod ativo")
        return
    _api("DELETE", f"/pods/{pod['id']}")
    print(f"pod apagado: {pod['id']}")


def watch(every_min: float = 5.0) -> None:
    """Acompanha até o job terminar; busca os resultados; confirma que o pod sumiu."""
    while True:
        state = status(verbose=False)
        camp = state.get("campanha") or {}
        print(f"[{time.strftime('%H:%M')}] pod={state['pod']} US$ {state['custo_ate_agora_usd']} "
              f"feitos={camp.get('done')}/{camp.get('total')} eta={camp.get('eta_h')} h "
              f"fim={state.get('job_done')}", flush=True)
        if state["pod"] != "RUNNING":
            print("o pod não está rodando: nada a buscar por SSH")
            return
        if state.get("job_done"):
            fetch()
            time.sleep(60)
            if _find_pod():
                print("o pod ainda existe depois da busca: apagando")
                down()
            return
        time.sleep(every_min * 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p_up = sub.add_parser("up")
    p_up.add_argument("campaign")
    p_up.add_argument("--budget", type=float, required=True, help="teto em US$ para este pod")
    p_up.add_argument("--parallel", type=int, default=3)
    p_up.add_argument("--grace", type=float, default=45, help="minutos esperando a busca no fim")
    for name in ("status", "fetch", "down"):
        sub.add_parser(name)
    p_watch = sub.add_parser("watch")
    p_watch.add_argument("--every", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        if args.action == "up":
            up(args.campaign, args.budget, args.parallel, args.grace)
        elif args.action == "watch":
            watch(args.every)
        else:
            {"status": status, "fetch": fetch, "down": down}[args.action]()
    except PodError as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
