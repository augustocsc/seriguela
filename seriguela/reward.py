"""Recompensa: de uma expressão (texto) para um ``Eval`` com R² e recompensa.

A recompensa é uma função monótona do ajuste no TREINO, sem degraus:

* ``r2``    -> ``clip(R², 0, 1)``
* ``nrmse`` -> ``1 / (1 + NRMSE)`` (a do DSR; dá sinal mesmo com R² < 0)

Expressões inválidas recebem ``invalid_reward`` e ENTRAM na atualização da política
(na v2 eram filtradas antes da loss; e R² em [0, 0.5) valia 0, uma zona morta).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .data import Problem
from .expr import Expression, InvalidExpression, complexity, r2_score


@dataclass(frozen=True)
class Eval:
    expression: str
    valid: bool
    r2: float            # no treino; -inf se inválida
    reward: float
    complexity: int
    error: str = ""      # "", "parse", "variables", "nan", "unterminated"
    constants: tuple = ()


class Evaluator:
    """Avalia expressões sobre o treino de um problema, com cache por texto."""

    def __init__(self, problem: Problem, reward: str = "r2", const_mode: str = "one",
                 invalid_reward: float = 0.0, is_prefix: bool = False):
        if reward not in ("r2", "nrmse"):
            raise ValueError(f"recompensa desconhecida: {reward}")
        if const_mode not in ("one", "fit"):
            raise ValueError(f"const_mode desconhecido: {const_mode}")
        self.problem = problem
        self.reward = reward
        self.const_mode = const_mode
        self.invalid_reward = invalid_reward
        self.is_prefix = is_prefix
        self._y_std = float(np.std(problem.y_train)) or 1.0
        self._cache: Dict[str, Eval] = {}
        self.calls = 0          # pedidos (com repetição)
        self.unique = 0         # avaliações de fato executadas

    def _invalid(self, text: str, error: str) -> Eval:
        return Eval(text, False, float("-inf"), self.invalid_reward,
                    complexity(text, self.is_prefix), error)

    def _evaluate(self, text: str) -> Eval:
        try:
            expr = Expression(text, self.is_prefix)
        except InvalidExpression:
            return self._invalid(text, "parse")
        if expr.max_var > self.problem.n_vars:
            return self._invalid(text, "variables")
        X, y = self.problem.x_train, self.problem.y_train
        constants = expr.fit_constants(X, y) if self.const_mode == "fit" else np.ones(expr.n_const)
        pred = expr.evaluate(X, constants)
        if not np.all(np.isfinite(pred)):
            return self._invalid(text, "nan")
        r2 = r2_score(y, pred)
        if self.reward == "r2":
            value = float(np.clip(r2, 0.0, 1.0))
        else:
            value = float(1.0 / (1.0 + np.sqrt(np.mean((y - pred) ** 2)) / self._y_std))
        return Eval(text, True, r2, value, complexity(text, self.is_prefix), "",
                    tuple(float(c) for c in constants))

    def __call__(self, text: Optional[str]) -> Eval:
        self.calls += 1
        if text is None:
            return self._invalid("", "unterminated")
        cached = self._cache.get(text)
        if cached is None:
            cached = self._evaluate(text)
            self._cache[text] = cached
            self.unique += 1
        return cached

    def test_r2(self, ev: Eval) -> Optional[float]:
        """R² no TESTE com as constantes escolhidas no treino. Chamado uma vez, no fim."""
        if not ev.valid:
            return None
        pred = Expression(ev.expression, self.is_prefix).evaluate(self.problem.x_test, ev.constants)
        value = r2_score(self.problem.y_test, pred)
        return value if np.isfinite(value) else None
