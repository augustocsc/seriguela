"""Lança a Fase 4.2 no RunPod: pod + bootstrap (reusa launch_pilot) + N daemons
concorrentes da fila sobre queue_phase4_{a,b,c}.yaml (NO_GIT=1, modo VM).

Os resultados ficam em results/phase_4/ NO POD (NO_GIT=1 não pusha) — colete com
--collect (scp) e termine com --terminate. O pod NÃO se auto-termina.

Pré-requisito: commitar+pushar queue_phase4_*.yaml ANTES (o bootstrap faz git pull).

Uso (máquina local, raiz do repo):
    python runpod/launch_phase4.py --splits 3      # cria pod, bootstrap, 3 daemons
    python runpod/launch_phase4.py --status
    python runpod/launch_phase4.py --collect        # scp results/phase_4 -> local
    python runpod/launch_phase4.py --terminate
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# reusa toda a maquinaria provada do piloto
import importlib.util
_LP = Path(__file__).resolve().parent / "launch_pilot.py"
_spec = importlib.util.spec_from_file_location("launch_pilot", _LP)
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)

lp.POD_NAME = "augusto-seriguela-phase4"      # nome distinto do piloto
POD_NAME = lp.POD_NAME
REPO = lp.REPO
INFO_FILE = REPO / "runpod" / "pod_info_phase4.json"


def _info():
    return json.loads(INFO_FILE.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=6,
                    help="nº de daemons concorrentes (filas queue_phase4_a..). "
                         "6 medido como ótimo (2026-06-26): feynman é CPU-bound "
                         "(eval simbólico single-thread), pod tem 64 vCPUs e a "
                         "GPU usa só ~6.4GB/run -> 6 cabem em 48GB e exploram a "
                         "CPU/GPU ociosa (~2x vs 3). Teto VRAM ~7.")
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    args = ap.parse_args()

    print("launch_phase4 — RunPod (reusa launch_pilot)")
    key = lp.read_key()

    if args.terminate:
        pod = lp.find_existing(key)
        if not pod:
            print("Nenhum pod ativo:", POD_NAME); return
        lp.rest(key, "DELETE", f"/pods/{pod['id']}")
        print("Pod terminado:", pod["id"]); return

    if args.collect:
        info = _info()
        ip, port = info["ip"], info["port"]
        dest = REPO / "results"
        print(f"Coletando results/phase_4 de {ip}:{port} ...")
        rc = subprocess.run(
            ["scp", *lp.SSH_OPTS, "-i", str(lp.SSH_KEY), "-P", str(port), "-r",
             f"root@{ip}:seriguela/results/phase_4", str(dest)]).returncode
        print("scp OK" if rc == 0 else f"scp rc={rc} (talvez ainda sem results)")
        return

    pod = lp.find_existing(key)
    if args.status:
        if not pod:
            print("Nenhum pod ativo:", POD_NAME); return
        info = _info()
        r = lp.ssh(info["ip"], info["port"],
                   "for f in /root/seriguela/phase4_*.log; do echo \"== $f ==\"; "
                   "tail -4 \"$f\"; done; echo '--- filas ---'; "
                   "grep -c 'status: done' /root/seriguela/experiments/queue_phase4_*.yaml 2>/dev/null; "
                   "echo '--- results ---'; "
                   "ls /root/seriguela/results/phase_4 2>/dev/null | wc -l")
        print(r.stdout or r.stderr); return

    # --- criar/reusar pod + bootstrap ---
    if pod:
        print(f"Pod reutilizado: {pod['id']} (${pod.get('costPerHr')}/h)")
        pod_id = pod["id"]
    else:
        pubkey = lp.ensure_ssh_key()
        dep = lp.deploy(key, pubkey)
        if not dep or not dep.get("id"):
            sys.exit("ERRO: nenhuma GPU deployou — ver mensagens da API acima")
        pod_id = dep["id"]

    ip, port, cost = lp.wait_ssh_endpoint(key, pod_id)
    INFO_FILE.write_text(json.dumps(
        {"pod_id": pod_id, "ip": ip, "port": port, "cost_per_hr": cost}, indent=2),
        encoding="utf-8")
    print(f"SSH: root@{ip} -p {port}  (${cost}/h)")

    for _ in range(30):
        if lp.ssh(ip, port, "echo ok").returncode == 0:
            break
        time.sleep(10)
    else:
        sys.exit("ERRO: SSH não respondeu")

    print("\n== bootstrap (clone + venv + deps + smoke) ==", flush=True)
    lp.ssh_stream(ip, port, lp.BOOTSTRAP, timeout=2400)
    r = lp.ssh(ip, port, "ls /root/READY /root/SMOKE_FAILED 2>/dev/null")
    if "READY" not in r.stdout:
        sys.exit("ERRO: bootstrap não chegou a READY (ver /root/bootstrap.log)")
    print("Bootstrap OK.")

    # garante que as filas da Fase 4 estão no pod (git pull do bootstrap já traz,
    # mas re-scp evita clone stale)
    print("== enviando filas Fase 4 ==", flush=True)
    for x in [chr(ord('a') + i) for i in range(args.splits)]:
        subprocess.run(["scp", *lp.SSH_OPTS, "-i", str(lp.SSH_KEY), "-P", str(port),
                        str(REPO / "experiments" / f"queue_phase4_{x}.yaml"),
                        f"root@{ip}:/root/seriguela/experiments/"], check=True)

    print(f"== iniciando {args.splits} daemons concorrentes ==", flush=True)
    for x in [chr(ord('a') + i) for i in range(args.splits)]:
        cmd = (f"cd /root/seriguela && source .venv/bin/activate && "
               f"NO_GIT=1 nohup python experiments/run_queue_file.py "
               f"experiments/queue_phase4_{x}.yaml --max-hours {args.max_hours} "
               f"> phase4_{x}.log 2>&1 & echo DAEMON_{x}_PID $!")
        r = lp.ssh(ip, port, cmd)
        print(" ", (r.stdout or r.stderr).strip())

    print(f"\nFASE 4 RODANDO ({args.splits} daemons). Acompanhe/colete:")
    print("  python runpod/launch_phase4.py --status")
    print("  python runpod/launch_phase4.py --collect      # scp results -> local")
    print("  python runpod/launch_phase4.py --terminate    # SEMPRE ao final")


if __name__ == "__main__":
    main()
