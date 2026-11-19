"""Vantagens: de recompensas por amostra para pesos da atualização.

Com um único prompt por problema, o "grupo" do GRPO é o lote inteiro; a vantagem
do GRPO (z-score no grupo) e a do PPO sem crítico (recompensa menos linha de base,
normalizada no lote) são então a MESMA função. ``zscore`` serve às duas.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

_EPS = 1e-8


def zscore(rewards: np.ndarray) -> np.ndarray:
    """(r - média) / desvio. Recompensas todas iguais -> vantagens zero."""
    r = np.asarray(rewards, dtype=np.float64)
    std = r.std()
    return np.zeros_like(r) if std < _EPS else (r - r.mean()) / std


def centered_rank(rewards: np.ndarray) -> np.ndarray:
    """Postos centrados e padronizados; empates recebem o posto médio (logo, o mesmo valor)."""
    r = np.asarray(rewards, dtype=np.float64)
    order = np.argsort(r, kind="stable")
    ranks = np.empty(len(r), dtype=np.float64)
    ranks[order] = np.arange(len(r), dtype=np.float64)
    for value in np.unique(r):                      # posto médio nos empates
        tied = r == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return zscore(ranks)


def risk_seeking(rewards: np.ndarray, epsilon: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Gradiente de política que busca risco (DSR): só o quantil superior atualiza.

    Devolve ``(vantagens, mantidas)``; a vantagem é ``r - r_eps`` nas amostras com
    ``r >= r_eps`` e zero nas demais.
    """
    r = np.asarray(rewards, dtype=np.float64)
    threshold = np.quantile(r, 1.0 - epsilon)
    keep = r >= threshold
    return np.where(keep, r - threshold, 0.0), keep


def compute(kind: str, rewards: np.ndarray, epsilon: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Despacha pelo nome. Devolve ``(vantagens, mantidas)``."""
    everyone = np.ones(len(rewards), dtype=bool)
    if kind == "zscore":
        return zscore(rewards), everyone
    if kind == "rank":
        return centered_rank(rewards), everyone
    if kind == "risk":
        return risk_seeking(rewards, epsilon)
    raise ValueError(f"vantagem desconhecida: {kind}")
