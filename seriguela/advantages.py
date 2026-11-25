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


def compute_with_replay(kind: str, rewards: np.ndarray, n_fresh: int,
                        epsilon: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula vantagens usando APENAS o lote novo como referência, sem que o replay
    distorça a média/desvio, os postos ou o quantil de risco das amostras novas.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if n_fresh >= len(r):
        return compute(kind, r, epsilon)
    r_fresh, r_rep = r[:n_fresh], r[n_fresh:]
    if kind == "zscore":
        mu, std = r_fresh.mean(), r_fresh.std()
        if std < _EPS:
            std_all = r.std()
            adv_fresh = np.zeros_like(r_fresh)
            adv_rep = np.zeros_like(r_rep) if std_all < _EPS else (r_rep - mu) / std_all
        else:
            adv_fresh = (r_fresh - mu) / std
            adv_rep = (r_rep - mu) / std
        return np.concatenate([adv_fresh, adv_rep]), np.ones(len(r), dtype=bool)
    if kind == "rank":
        adv_fresh = centered_rank(r_fresh)
        s = np.sort(r_fresh)
        left = np.searchsorted(s, r_rep, side="left")
        right = np.searchsorted(s, r_rep, side="right")
        # posição média equivalente na escala [0, n_fresh - 1]
        rep_ranks = 0.5 * (left + np.maximum(right - 1, left))
        raw_fresh = np.arange(len(r_fresh), dtype=np.float64)
        mu, std = raw_fresh.mean(), raw_fresh.std()
        adv_rep = np.zeros_like(r_rep) if std < _EPS else (rep_ranks - mu) / std
        return np.concatenate([adv_fresh, adv_rep]), np.ones(len(r), dtype=bool)
    if kind == "risk":
        threshold = np.quantile(r_fresh, 1.0 - epsilon)
        keep_fresh = r_fresh >= threshold
        keep_rep = r_rep >= threshold
        adv_fresh = np.where(keep_fresh, r_fresh - threshold, 0.0)
        adv_rep = np.where(keep_rep, r_rep - threshold, 0.0)
        return np.concatenate([adv_fresh, adv_rep]), np.concatenate([keep_fresh, keep_rep])
    raise ValueError(f"vantagem desconhecida: {kind}")
