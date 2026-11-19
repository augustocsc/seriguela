"""Equivalência simbólica no critério do SRBench (La Cava et al., 2021, def. 4.1).

Uma predição é solução se, depois de simplificada, difere do alvo por uma constante
aditiva, ou se a razão alvo/predição é uma constante não nula -- e a predição não é
ela própria uma constante. É um critério mais forte que ``R² >= 0,999``; os dois são
relatados em separado.

Duas etapas:

1. triagem numérica: se nem ``alvo - pred`` nem ``alvo / pred`` são constantes em
   alguns pontos, a resposta é ``False`` sem chamar o sympy;
2. confirmação com ``sympy.simplify``, num processo com prazo (pode não terminar).
"""
from __future__ import annotations

import multiprocessing as mp
import re
from typing import Optional, Sequence

import numpy as np

from .expr import reference_values

_CONST = re.compile(r"\bC\b")
_VAR = re.compile(r"\bx_(\d+)\b")


def _substitute_constants(text: str, constants: Optional[Sequence[float]]) -> str:
    values = iter(constants) if constants is not None else None
    return _CONST.sub(lambda _: repr(float(next(values))) if values is not None else "1", text)


def _plausible(pred: str, truth: str, points: Optional[np.ndarray], tol: float = 1e-6) -> bool:
    """``False`` só quando os números já excluem diferença e razão constantes."""
    n_vars = max([int(i) for i in _VAR.findall(pred + " " + truth)] or [1])
    if points is None or points.shape[1] < n_vars:
        points = np.random.default_rng(0).uniform(0.5, 1.5, (16, n_vars))
    try:
        p, t = reference_values(pred, points[:16]), reference_values(truth, points[:16])
    except Exception:
        return True
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 4:
        return True                                   # sem pontos para decidir
    p, t = p[ok], t[ok]
    scale = max(float(np.max(np.abs(t))), 1.0)
    if np.ptp(p) <= tol * scale:                      # predição constante nos pontos
        return bool(np.ptp(t) <= tol * scale)
    if np.ptp(t - p) <= tol * scale:
        return True
    nonzero = np.abs(p) > 1e-12
    if nonzero.all():
        ratio = t / p
        return bool(np.ptp(ratio) <= tol * max(float(np.max(np.abs(ratio))), 1.0) and abs(ratio.mean()) > tol)
    return False


def _decide(pred: str, truth: str) -> bool:
    import sympy
    p, t = sympy.sympify(pred), sympy.sympify(truth)
    if not p.free_symbols:                            # modelo constante não conta
        return False
    difference = sympy.simplify(t - p)
    if difference.is_constant():
        return True
    ratio = sympy.simplify(t / p)
    return bool(ratio.is_constant() and ratio != 0)


def _worker(pred: str, truth: str, queue) -> None:
    try:
        queue.put(_decide(pred, truth))
    except Exception:
        queue.put(False)


def symbolic_match(pred: str, truth: str, constants: Optional[Sequence[float]] = None,
                   points: Optional[np.ndarray] = None, timeout: float = 60.0) -> Optional[bool]:
    """``True``/``False``, ou ``None`` se o sympy não decidiu no prazo."""
    try:
        pred = _substitute_constants(pred, constants)
    except StopIteration:
        return False
    if not _plausible(pred, truth, points):
        return False
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(pred, truth, queue), daemon=True)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    return bool(queue.get()) if not queue.empty() else None
