"""Campanha: fila de runs a partir de um arquivo, vários processos por GPU, teto de horas.

    python -m seriguela.campaign plan   campaigns/g1_timing.json
    python -m seriguela.campaign work   campaigns/g1_timing.json --parallel 3 --max_hours 1.5
    python -m seriguela.campaign status campaigns/g1_timing.json

O arquivo da campanha (JSON) declara células; cada célula expande em problemas × algos ×
replays × seeds. ``bon`` só existe sem replay. Um run já gravado conta como feito, então
``work`` retoma de onde parou.

``work`` para de lançar runs e mata os que estão rodando quando:
- um run falha (tripwire = código 2, ou qualquer outro erro): há um bug, não adianta gastar;
- o teto de horas acaba (os runs interrompidos não gravam nada e voltam à fila).
O estado vai para ``<out>/campaign_status.json`` a cada poucos segundos (é o que o
controle do pod lê). Código de saída: 0 se a fila terminou, 1 caso contrário.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .trainer import PRESETS, RunConfig

REPLAYS = ("none", "elite", "random")
_FIELDS = {f.name for f in dataclasses.fields(RunConfig)}


class CampaignError(ValueError):
    """Arquivo de campanha inválido."""


@dataclasses.dataclass(frozen=True)
class Task:
    config: Dict
    out: Path

    @property
    def cfg(self) -> RunConfig:
        return RunConfig(**self.config)

    @property
    def id(self) -> str:
        c = self.cfg
        return f"{c.problem}/{c.arm}/seed_{c.seed}"

    @property
    def path(self) -> Path:
        c = self.cfg
        return self.out / c.problem / c.arm / f"seed_{c.seed}.json"

    @property
    def failed_path(self) -> Path:
        return self.path.with_name(f"seed_{self.cfg.seed}.failed.json")

    def command(self, allow_dirty: bool = False) -> List[str]:
        cmd = [sys.executable, "-m", "seriguela.run"]
        for key, value in self.config.items():
            if value is not None:
                cmd += [f"--{key}", str(value)]
        cmd += ["--out", str(self.out)]
        return cmd + (["--allow_dirty"] if allow_dirty else [])


def load(path) -> Dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    spec.pop("description", None)
    for key in ("name", "cells"):
        if key not in spec:
            raise CampaignError(f"campanha sem '{key}'")
    spec.setdefault("common", {})
    spec.setdefault("out", f"results/v3/{spec['name']}")
    return spec


def expand(spec: Dict) -> List[Task]:
    """Lista de runs da campanha, na ordem do arquivo. Rejeita campos e braços desconhecidos."""
    out = Path(spec["out"])
    tasks, seen = [], set()
    for n, cell in enumerate(spec["cells"]):
        unknown = set(cell) - {"problems", "algos", "replays", "seeds", "overrides"}
        if unknown:
            raise CampaignError(f"célula {n}: chaves desconhecidas {sorted(unknown)}")
        overrides = {**spec["common"], **cell.get("overrides", {})}
        bad = set(overrides) - _FIELDS
        if bad or {"problem", "algo", "replay", "seed"} & set(overrides):
            raise CampaignError(f"célula {n}: campos inválidos em overrides {sorted(bad) or 'de identidade'}")
        for algo in cell["algos"]:
            if algo not in PRESETS:
                raise CampaignError(f"célula {n}: algo desconhecido {algo}")
        for replay in cell.get("replays", ["none"]):
            if replay not in REPLAYS:
                raise CampaignError(f"célula {n}: replay desconhecido {replay}")
        for problem in cell["problems"]:
            for algo in cell["algos"]:
                for replay in cell.get("replays", ["none"]):
                    if algo == "bon" and replay != "none":
                        continue
                    for seed in cell["seeds"]:
                        config = {"problem": problem, "algo": algo, "replay": replay, "seed": int(seed),
                                  **overrides}
                        task = Task(config, out)
                        if task.id in seen:
                            raise CampaignError(f"run repetido: {task.id}")
                        seen.add(task.id)
                        tasks.append(task)
    return tasks


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def status_path(spec: Dict) -> Path:
    return Path(spec["out"]) / "campaign_status.json"


def work(spec: Dict, parallel: int = 1, max_hours: float = 1.0, allow_dirty: bool = False,
         poll: float = 5.0, command: Optional[Callable[[Task], Sequence[str]]] = None,
         clock: Callable[[], float] = time.time) -> Dict:
    """Roda a fila. ``command`` troca o comando de cada run (testes)."""
    tasks = expand(spec)
    out = Path(spec["out"])
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        if task.failed_path.exists():
            raise CampaignError(f"há falha registrada em {task.failed_path}: conserte o bug, "
                                "apague o arquivo e rode de novo")
    queue = [t for t in tasks if not t.path.exists()]
    started = clock()
    deadline = started + max_hours * 3600
    running: Dict[str, tuple] = {}
    durations: List[float] = []
    done = len(tasks) - len(queue)
    failures: List[Dict] = []
    reason = None

    def snapshot(final=False):
        elapsed = clock() - started
        mean = sum(durations) / len(durations) if durations else None
        left = len(queue) + len(running)
        eta = mean * left / max(parallel, 1) / 3600 if mean else None
        _write_json(status_path(spec), {
            "name": spec["name"], "updated": _now(), "elapsed_h": round(elapsed / 3600, 4),
            "max_hours": max_hours, "parallel": parallel, "total": len(tasks), "done": done,
            "running": sorted(running), "pending": len(queue), "failures": failures,
            "mean_run_seconds": mean, "eta_h": eta, "finished": final, "reason": reason})

    while queue or running:
        if clock() > deadline:
            reason = "deadline"
            break
        while queue and len(running) < parallel and not failures:
            task = queue.pop(0)
            cmd = list(command(task) if command else task.command(allow_dirty))
            log = open(logs / (task.id.replace("/", "__") + ".log"), "w", encoding="utf-8")
            running[task.id] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, clock(), task)
        for tid, (proc, log, t0, task) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[tid]
            if code == 0 and task.path.exists():
                done += 1
                durations.append(clock() - t0)
            else:
                failures.append({"run": tid, "exit_code": code,
                                 "tripwire": task.failed_path.exists(), "log": str(log.name)})
        if failures:
            reason = "failure"
            break
        snapshot()
        if queue or running:
            time.sleep(poll)
    for tid, (proc, log, _, _) in running.items():      # para o que sobrou
        proc.kill()
        proc.wait()
        log.close()
    reason = reason or "complete"
    running.clear()
    snapshot(final=True)
    return json.loads(status_path(spec).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["plan", "work", "status"])
    parser.add_argument("campaign")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max_hours", type=float, default=1.0)
    parser.add_argument("--allow_dirty", action="store_true")
    args = parser.parse_args(argv)
    spec = load(args.campaign)
    if args.action == "plan":
        tasks = expand(spec)
        pending = [t for t in tasks if not t.path.exists()]
        print(f"{spec['name']}: {len(tasks)} runs, {len(pending)} pendentes -> {spec['out']}")
        for t in pending[:20]:
            print("  ", t.id)
        if len(pending) > 20:
            print(f"   ... e mais {len(pending) - 20}")
        return 0
    if args.action == "status":
        path = status_path(spec)
        print(path.read_text(encoding="utf-8") if path.exists() else "sem status ainda")
        return 0
    status = work(spec, args.parallel, args.max_hours, args.allow_dirty)
    print(f"{status['reason']}: {status['done']}/{status['total']} feitos, "
          f"{len(status['failures'])} falha(s), {status['elapsed_h']:.2f} h")
    return 0 if status["reason"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
