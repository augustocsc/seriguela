"""Proveniência: o que é preciso para ligar um resultado ao código que o produziu."""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


class DirtyTreeError(RuntimeError):
    """Há mudanças não commitadas em arquivos rastreados."""


def git_state() -> Tuple[str, bool]:
    """``(commit, sujo)``. Arquivos não rastreados (ex.: resultados) não contam."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "HEAD") or "desconhecido", bool(git("status", "--porcelain", "--untracked-files=no"))


def config_hash(config: Dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def collect(config: Dict, data_hash: str, allow_dirty: bool = False) -> Dict:
    import numpy
    import peft
    import sympy
    import torch
    import transformers

    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise DirtyTreeError("árvore git suja: faça commit antes de um run de campanha "
                             "(ou use --allow_dirty em testes)")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "commit": commit, "dirty": dirty, "config_hash": config_hash(config), "data_hash": data_hash,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(), "gpu": gpu,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "numpy": numpy.__version__, "sympy": sympy.__version__},
    }
