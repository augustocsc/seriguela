"""Problemas de regressão simbólica: Nguyen (gerado) e SRBench/PMLB (arquivos).

Todo problema é devolvido já dividido em treino e teste. O teste só é lido pela
avaliação final (``trainer._finalize``); a busca nunca o vê.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import re
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "data" / "benchmarks"
PMLB_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv.gz"

# (equação, n_vars, domínio canônico (lo, hi, n), domínio usado na v2)
# Canônico = tabela usada pelo DSR (Petersen et al., 2021): amostragem uniforme.
_NGUYEN = {
    "nguyen_1": ("x_1**3 + x_1**2 + x_1", 1, (-1, 1, 20), (0, 2, 100)),
    "nguyen_2": ("x_1**4 + x_1**3 + x_1**2 + x_1", 1, (-1, 1, 20), (0, 2, 100)),
    "nguyen_3": ("x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1", 1, (-1, 1, 20), (0, 2, 100)),
    "nguyen_4": ("x_1**6 + x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1", 1, (-1, 1, 20), (0, 2, 100)),
    "nguyen_5": ("sin(x_1**2)*cos(x_1) - 1", 1, (-1, 1, 20), (0, 2, 100)),
    "nguyen_6": ("sin(x_1) + sin(x_1 + x_1**2)", 1, (-1, 1, 20), (0, 2, 100)),
    "nguyen_7": ("log(x_1 + 1) + log(x_1**2 + 1)", 1, (0, 2, 20), (0, 2, 100)),
    "nguyen_8": ("sqrt(x_1)", 1, (0, 4, 20), (0, 4, 100)),
    "nguyen_9": ("sin(x_1) + sin(x_2**2)", 2, (0, 1, 20), (0, 2, 100)),
    "nguyen_10": ("2*sin(x_1)*cos(x_2)", 2, (0, 1, 20), (0, 2, 100)),
    "nguyen_11": ("x_1**x_2", 2, (0, 1, 20), (0, 2, 100)),
    "nguyen_12": ("x_1**4 - x_1**3 + x_2**2/2 - x_2", 2, (0, 1, 20), (0, 2, 100)),
}
NGUYEN_PROBLEMS = tuple(_NGUYEN)
_NP = {"sin": np.sin, "cos": np.cos, "log": np.log, "sqrt": np.sqrt, "exp": np.exp}


@dataclass
class Problem:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    truth: Optional[str] = None          # equação-alvo em x_1..x_n, se conhecida
    meta: Dict = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return self.x_train.shape[1]

    def data_hash(self) -> str:
        digest = hashlib.sha256()
        for array in (self.x_train, self.y_train, self.x_test, self.y_test):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()[:16]


def _nguyen(name: str, seed: int, variant: str, test_scale: float) -> Problem:
    equation, n_vars, canonical, legacy = _NGUYEN[name]
    lo, hi, n = canonical if variant == "canonical" else legacy

    def draw(rng, low, high, count):
        x = rng.uniform(low, high, (count, n_vars))
        env = {f"x_{i + 1}": x[:, i] for i in range(n_vars)}
        return x, eval(equation, {"__builtins__": {}}, {**_NP, **env})

    x_train, y_train = draw(np.random.default_rng([seed, 0]), lo, hi, n)
    # Teste: mesma função, sementes diferentes; test_scale > 1 alarga o intervalo
    # (extrapolação) em torno do centro, limitado ao domínio da função.
    center, half = (lo + hi) / 2, (hi - lo) / 2 * test_scale
    t_lo, t_hi = center - half, center + half
    if lo >= 0:  # log/sqrt/x**y: não sair do domínio positivo
        t_lo = max(t_lo, 0.0)
    x_test, y_test = draw(np.random.default_rng([seed, 1]), t_lo, t_hi, max(n, 100))
    return Problem(name, x_train, y_train, x_test, y_test, truth=equation,
                   meta={"family": "nguyen", "variant": variant, "domain": [lo, hi, n],
                         "test_domain": [t_lo, t_hi]})


def _download(name: str, csv_path: Path) -> None:
    import pandas as pd
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    gz_path = csv_path.with_suffix(".tsv.gz")
    urllib.request.urlretrieve(PMLB_URL.format(name=name), gz_path)
    with gzip.open(gz_path, "rb") as src, open(csv_path.with_suffix(".tsv"), "wb") as dst:
        shutil.copyfileobj(src, dst)
    frame = pd.read_csv(csv_path.with_suffix(".tsv"), sep="\t")
    frame.to_csv(csv_path, index=False)
    gz_path.unlink()
    csv_path.with_suffix(".tsv").unlink()


def _normalize_truth(equation: str, columns) -> str:
    """Padroniza nomes de funções para SymPy e renomeia colunas para ``x_1..x_n``."""
    equation = re.sub(r"\barcsin\b", "asin", equation)
    equation = re.sub(r"\barccos\b", "acos", equation)
    equation = re.sub(r"\barctan\b", "atan", equation)
    equation = re.sub(r"\bln\b", "log", equation)
    for index, column in sorted(enumerate(columns), key=lambda kv: -len(kv[1])):
        equation = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])", f"x_{index + 1}", equation)
    return equation


def _meta_truth(name: str, family: str, columns) -> Optional[str]:
    """Lê a fórmula oficial da descrição PMLB (cobre Feynman e Strogatz)."""
    import json
    meta_path = BENCH_DIR / family / f"{name}.meta.json"
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    desc = data.get("pmlb_metadata", {}).get("description", "")
    for line in desc.splitlines():
        s = line.strip()
        if "=" in s and not s.startswith("http"):
            rhs = s.split("=", 1)[1].strip()
            if rhs:
                return _normalize_truth(rhs, columns)
    return None


def _feynman_truth(name: str, columns) -> Optional[str]:
    path = BENCH_DIR / "feynman_equations.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("feynman_equations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, "FEYNMAN_EQUATIONS", {}).get(name)
    if not entry:
        return None
    return _normalize_truth(entry["equation"], columns)


def srbench_truth(name: str, family: str, columns) -> Optional[str]:
    """Equação-alvo em ``x_1..x_n``: prioriza metadado oficial PMLB, com fallback."""
    truth = _meta_truth(name, family, columns)
    if truth is not None:
        return truth
    return _feynman_truth(name, columns) if family == "feynman" else None


def _srbench(name: str, seed: int, max_train: int, noise: float, download: bool) -> Problem:
    import pandas as pd
    family = "feynman" if name.startswith("feynman") else "strogatz"
    csv_path = BENCH_DIR / family / f"{name}.csv"
    if not csv_path.exists():
        if not download:
            raise FileNotFoundError(f"{csv_path} ausente (rode com download=True)")
        _download(name, csv_path)
    frame = pd.read_csv(csv_path)
    features = [column for column in frame.columns if column != "target"]
    X = frame[features].to_numpy(dtype=np.float64)
    y = frame["target"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng([seed, 2])
    order = rng.permutation(len(y))
    n_test = max(1, int(len(y) * 0.25))             # 75/25, como no SRBench
    test_idx, train_idx = order[:n_test], order[n_test:]
    if max_train and len(train_idx) > max_train:
        train_idx = train_idx[:max_train]
    y_train = y[train_idx].copy()
    if noise > 0:                                    # mesma definição do SRBench
        y_train += rng.normal(0, noise * np.sqrt(np.mean(np.square(y_train))), len(y_train))
    truth = srbench_truth(name, family, features)
    return Problem(name, X[train_idx], y_train, X[test_idx], y[test_idx], truth=truth,
                   meta={"family": family, "columns": features, "noise": noise,
                         "n_rows": int(len(y)), "max_train": max_train})


def load_problem(name: str, seed: int, *, nguyen_variant: str = "canonical",
                 test_scale: float = 1.0, max_train: int = 10000, noise: float = 0.0,
                 download: bool = True) -> Problem:
    """Carrega um problema pelo nome (``nguyen_5``, ``feynman_I_12_1``, ``strogatz_lv1``)."""
    if name in _NGUYEN:
        if nguyen_variant not in ("canonical", "v2"):
            raise ValueError(f"variante Nguyen desconhecida: {nguyen_variant}")
        return _nguyen(name, seed, nguyen_variant, test_scale)
    if name.startswith(("feynman", "strogatz")):
        return _srbench(name, seed, max_train, noise, download)
    raise ValueError(f"problema desconhecido: {name}")
