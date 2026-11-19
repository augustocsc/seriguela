"""End-to-end RunPod pilot launcher (stdlib-only, idempotent) — REST API v1, Bearer auth.

Creates (or reuses) a community-cloud GPU pod, bootstraps the repo + venv on it,
runs the mandatory smoke test there, uploads the timing harness, and starts the
pilot under nohup. Never prints the API key.

Usage (local machine, repo root):
    python runpod/launch_pilot.py            # full flow
    python runpod/launch_pilot.py --status   # just show pod + pilot status
    python runpod/launch_pilot.py --quick    # pass --quick to the pilot harness
    python runpod/launch_pilot.py --terminate

GPU preference (any available): NVIDIA A40 (48GB) -> RTX 4090 -> RTX A5000.
Pod name: augusto-seriguela-pilot (guard against double-launch).
Pod info cached locally in runpod/pod_info.json for follow-up commands.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://rest.runpod.io/v1"
POD_NAME = "augusto-seriguela-pilot"
GPU_PREFS = [
    "NVIDIA A40",                    # 48GB — melhor $/run concorrente
    "NVIDIA RTX A6000",              # 48GB
    "NVIDIA GeForce RTX 3090",       # 24GB — muito barata
    "NVIDIA GeForce RTX 4090",       # 24GB — rápida
    "NVIDIA L40S",                   # 48GB
    "NVIDIA RTX A5000",              # 24GB
    "NVIDIA L4",                     # 24GB
]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
HOME = Path.home()
SSH_KEY = HOME / ".ssh" / "runpod_pilot"
REPO = Path(__file__).resolve().parents[1]
INFO_FILE = REPO / "runpod" / "pod_info.json"

BOOTSTRAP = r"""
exec &> >(tee /root/bootstrap.log)
set -ux
cd /root
rm -f /root/READY /root/SMOKE_FAILED
if [ ! -d seriguela ]; then git clone https://github.com/augustocsc/seriguela.git; fi
cd seriguela || { echo "CLONE_FAILED: repositório inacessível (privado? rede?)"; exit 1; }
git pull --ff-only || true
if [ ! -d .venv ]; then python -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
python experiments/test_smoke.py && touch /root/READY || touch /root/SMOKE_FAILED
ls /root/READY /root/SMOKE_FAILED 2>/dev/null
"""


def read_key() -> str:
    tokens = (HOME / ".tokens.txt").read_text(encoding="utf-8", errors="ignore")
    for line in tokens.splitlines():
        if line.strip().lower().startswith("runpod"):
            return line.split("=", 1)[1].strip()
    sys.exit("ERRO: linha 'runpod = <chave>' não encontrada em ~/.tokens.txt")


def rest(key: str, method: str, path: str, body: dict | None = None):
    """Call the RunPod REST API. Returns parsed JSON (or None for empty body).

    On HTTP errors, prints the response body (RunPod returns useful JSON errors)
    and returns None — callers decide whether that is fatal.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        print(f"[API {e.code}] {method} {path}\n{detail[:1000]}", file=sys.stderr)
        if e.code in (401, 403):
            print("DICA: verifique em runpod.io Settings -> API Keys se a chave tem "
                  "permissão 'All' ou Read & Write (chave restrita sem escopo dá 401/403).",
                  file=sys.stderr)
        return None


def ensure_ssh_key() -> str:
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(SSH_KEY)], check=True)
    return SSH_KEY.with_suffix(".pub").read_text().strip()


def list_pods(key: str) -> list:
    out = rest(key, "GET", "/pods")
    if out is None:
        return []
    return out if isinstance(out, list) else out.get("pods", [])


def find_existing(key: str):
    for pod in list_pods(key):
        if pod.get("name") == POD_NAME and pod.get("desiredStatus") in ("RUNNING", "PENDING"):
            return pod
    return None


def deploy(key: str, pubkey: str):
    """Try each cloud/GPU/disk combination until one machine accepts the pod."""
    for cloud in ("COMMUNITY", "SECURE"):
        for gpu in GPU_PREFS:
            for disk in (40, 30):
                print(f"Tentando deploy: {cloud} / {gpu} / disk {disk}GB ...", flush=True)
                body = {
                    "name": POD_NAME,
                    "imageName": IMAGE,
                    "cloudType": cloud,
                    "computeType": "GPU",
                    "gpuCount": 1,
                    "gpuTypeIds": [gpu],
                    "containerDiskInGb": disk,
                    "volumeInGb": 0,
                    "ports": ["22/tcp"],
                    "supportPublicIp": True,
                    "env": {"PUBLIC_KEY": pubkey},
                }
                out = rest(key, "POST", "/pods", body)
                if out and out.get("id"):
                    print(f"OK: {cloud} {gpu}, disk {disk}GB, ${out.get('costPerHr')}/h")
                    return out
                time.sleep(3)
    return None


def get_pod(key: str, pod_id: str):
    return rest(key, "GET", f"/pods/{pod_id}")


def extract_ssh(pod: dict):
    ip = pod.get("publicIp") or ""
    pm = pod.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    if ip and port:
        return ip, int(port)
    # fallback: GraphQL-style runtime.ports shape, if present
    for p in ((pod.get("runtime") or {}).get("ports") or []):
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], int(p["publicPort"])
    return None, None


def wait_ssh_endpoint(key: str, pod_id: str, timeout_s: int = 600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pod = get_pod(key, pod_id) or {}
        ip, port = extract_ssh(pod)
        if ip and port:
            return ip, port, pod.get("costPerHr")
        print(f"  aguardando endpoint SSH... (status={pod.get('desiredStatus')})", flush=True)
        time.sleep(20)
    sys.exit("ERRO: pod não expôs SSH em 10 min — verifique o console do RunPod")


# -F <vazio>: ignora ~/.ssh/config do usuário (um config corrompido — ex. BOM
# no início — derruba ssh/scp inteiros; visto em produção em 2026-06-10).
# Arquivo real em vez de os.devnull: no Windows, o ssh resolvido costuma ser o
# MSYS do Git, onde "nul" não existe ("Can't open user config file nul").
EMPTY_SSH_CONFIG = Path(__file__).resolve().parent / "ssh_config.empty"
SSH_OPTS = ["-F", str(EMPTY_SSH_CONFIG), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]


def ssh(ip: str, port: int, cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(SSH_KEY), "-p", str(port), f"root@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout)


def ssh_stream(ip: str, port: int, script: str, timeout: int) -> int:
    """Run a script on the pod with LIVE output in the local terminal.

    Sends raw bytes with LF endings — text mode on Windows would translate
    \n to \r\n and break bash on the Linux side.
    """
    payload = script.replace("\r\n", "\n").encode()
    p = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(SSH_KEY), "-p", str(port), f"root@{ip}", "bash -s"],
        input=payload, timeout=timeout)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("launch_pilot v2 — RunPod REST API (Bearer)")
    key = read_key()

    if args.terminate:
        pod = find_existing(key)
        if not pod:
            print("Nenhum pod ativo com nome", POD_NAME)
            return
        rest(key, "DELETE", f"/pods/{pod['id']}")
        print("Pod terminado:", pod["id"])
        return

    pod = find_existing(key)
    if args.status and not pod:
        # --status nunca deve criar pod (bug corrigido em 2026-06-10: o fluxo
        # antigo caía no deploy e subia um pod novo só para perguntar o status)
        print("Nenhum pod ativo com nome", POD_NAME)
        return
    if pod:
        print(f"Pod existente reutilizado: {pod['id']} (${pod.get('costPerHr')}/h)")
        pod_id = pod["id"]
    else:
        pubkey = ensure_ssh_key()
        dep = deploy(key, pubkey)
        if not dep or not dep.get("id"):
            sys.exit("ERRO: nenhuma combinação de GPU/disk deployou — ver mensagens da API acima")
        pod_id = dep["id"]

    ip, port, cost = wait_ssh_endpoint(key, pod_id)
    INFO_FILE.write_text(json.dumps({"pod_id": pod_id, "ip": ip, "port": port,
                                     "cost_per_hr": cost}, indent=2), encoding="utf-8")
    print(f"SSH: root@{ip} -p {port}  (${cost}/h) — salvo em runpod/pod_info.json")

    if args.status:
        r = ssh(ip, port,
                "echo '--- markers ---'; ls /root/READY /root/SMOKE_FAILED 2>/dev/null; "
                "echo '--- bootstrap.log (tail) ---'; tail -15 /root/bootstrap.log 2>/dev/null; "
                "echo '--- pilot.log (tail) ---'; tail -15 /root/seriguela/pilot.log 2>/dev/null; "
                "echo '--- timing_report.md ---'; "
                "tail -40 /root/seriguela/results/pilot_timing/timing_report.md 2>/dev/null")
        print(r.stdout or r.stderr)
        return

    for _ in range(30):
        r = ssh(ip, port, "echo ok")
        if r.returncode == 0 and "ok" in r.stdout:
            break
        time.sleep(10)
    else:
        sys.exit("ERRO: SSH não respondeu")

    print("\n== bootstrap (clone + venv + deps + smoke test) — ~8-15 min, saída ao vivo ==",
          flush=True)
    ssh_stream(ip, port, BOOTSTRAP, timeout=2400)
    r = ssh(ip, port, "ls /root/READY /root/SMOKE_FAILED 2>/dev/null")
    if "SMOKE_FAILED" in r.stdout:
        sys.exit("ERRO: smoke test FALHOU no pod — ver log acima (também em /root/bootstrap.log)")
    if "READY" not in r.stdout:
        sys.exit("ERRO: bootstrap não chegou ao marcador READY — ver log acima "
                 "(também em /root/bootstrap.log no pod)")
    print("Bootstrap OK (smoke test passou no pod).")

    print("== enviando harness ==", flush=True)
    subprocess.run(["scp", *SSH_OPTS, "-i", str(SSH_KEY), "-P", str(port),
                    str(REPO / "experiments" / "run_pilot_timing.py"),
                    f"root@{ip}:/root/seriguela/experiments/run_pilot_timing.py"], check=True)

    flag = "--quick" if args.quick else ""
    price = f"--price {cost}" if cost else ""
    r = ssh(ip, port, f"cd /root/seriguela && source .venv/bin/activate && "
                      f"nohup python experiments/run_pilot_timing.py {flag} {price} "
                      f"> pilot.log 2>&1 & echo PILOT_STARTED $!")
    print(r.stdout or r.stderr)
    if "PILOT_STARTED" not in r.stdout:
        sys.exit("ERRO: não confirmei o início do piloto")

    print("\nPILOTO RODANDO. Acompanhe com:")
    print("  python runpod/launch_pilot.py --status")
    print(f"  ssh -i {SSH_KEY} -p {port} root@{ip} 'tail -f seriguela/pilot.log'")
    print("Ao final, colete o relatório e TERMINE o pod:")
    print(f"  scp -i {SSH_KEY} -P {port} root@{ip}:seriguela/results/pilot_timing/timing_report.* .")
    print("  python runpod/launch_pilot.py --terminate")


if __name__ == "__main__":
    main()
