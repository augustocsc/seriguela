"""Expressões simbólicas: parse seguro, avaliação numérica e constantes ``C``.

Cada ocorrência de ``C`` é um parâmetro próprio. No protocolo padrão todas valem 1
(``const_mode="one"``); ``fit_constants`` ajusta-as por L-BFGS-B.

Há dois avaliadores de propósito:

* ``Expression.evaluate`` -- rápido (numpy), usado na recompensa;
* ``reference_values`` -- lento e independente (sympy -> math), usado só para
  conferir o primeiro. O bug do expoente-C da v2 (``x**C`` virava ``x**2``) teria
  sido pego por essa conferência no primeiro passo.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

import numpy as np
import sympy

MAX_COMPLEXITY = 100   # tokens; alvos reais têm ~5-15
MAX_EXPONENT = 100     # expoente numérico literal

_ALLOWED = re.compile(r"^[0-9a-zA-Z_+\-*/().,^\s]*$")
_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[+\-*/^()]")
_VAR = re.compile(r"\bx_(\d+)\b")
_CONST = re.compile(r"\bC\b")
_INT = re.compile(r"(?<![\w.])\d+(?![\w.])")
_UNARY = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "asin"}
_BINARY = {"+", "-", "*", "/", "**", "^"}
_NP_FUNCS = {
    "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "asin": np.arcsin, "abs": np.abs, "Abs": np.abs,
}
_MATH_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "asin": math.asin, "Abs": abs, "pi": math.pi, "E": math.e,
}


class InvalidExpression(ValueError):
    """A expressão não pode ser interpretada com segurança."""


def complexity(text: str, is_prefix: bool = False) -> int:
    """Número de tokens (operadores, operandos e parênteses)."""
    return len(text.split()) if is_prefix else len(_TOKEN.findall(text))


def _check_guards(text: str) -> None:
    """Rejeita, antes do sympy, entradas que estouram memória ou travam o numpy."""
    if not text.strip():
        raise InvalidExpression("vazia")
    if "__" in text or not _ALLOWED.match(text):
        raise InvalidExpression("caractere não permitido")
    if complexity(text) > MAX_COMPLEXITY:
        raise InvalidExpression("complexidade acima do teto")
    for exponent in re.findall(r"\*\*\s*\(?\s*-?(\d+(?:\.\d+)?)", text):
        if float(exponent) > MAX_EXPONENT:
            raise InvalidExpression("expoente literal grande")
    # expoente que é ele próprio uma potência: x**(9**9) ou x**2**3
    if re.search(r"\*\*\s*\([^)]*\*\*", text) or re.search(r"\*\*\s*\d+(?:\.\d+)?\s*\*\*", text):
        raise InvalidExpression("torre de potências")


def prefix_to_infix(tokens: Sequence[str]) -> str:
    """``['*', 'x_1', '+', 'x_2', 'C']`` -> ``(x_1)*((x_2)+(C))``."""
    stack: List[str] = []
    for token in reversed(list(tokens)):
        if token in _UNARY:
            if not stack:
                raise InvalidExpression(f"falta operando para {token}")
            stack.append(f"{token}({stack.pop()})")
        elif token in _BINARY:
            if len(stack) < 2:
                raise InvalidExpression(f"faltam operandos para {token}")
            left, right = stack.pop(), stack.pop()
            op = "**" if token == "^" else token
            stack.append(f"({left}){op}({right})")
        else:
            stack.append(token)
    if len(stack) != 1:
        raise InvalidExpression("prefixo mal formado")
    return stack[0]


class Expression:
    """Expressão em ``x_1..x_n`` e constantes ``C``, avaliável sobre uma matriz ``X``."""

    def __init__(self, text: str, is_prefix: bool = False):
        self.text = text
        source = prefix_to_infix(text.replace("^", "**").split()) if is_prefix else text
        _check_guards(source)
        try:
            self.sympy_expr = sympy.sympify(source, evaluate=False)
        except Exception as exc:  # sympy levanta de tudo
            raise InvalidExpression(f"parse falhou: {exc}") from exc
        if not isinstance(self.sympy_expr, sympy.Basic):
            raise InvalidExpression("não é uma expressão")
        code = str(self.sympy_expr)
        self.variables = sorted({int(i) for i in _VAR.findall(code)})
        self.max_var = max(self.variables, default=0)
        self.n_const = len(_CONST.findall(code))
        # Literais inteiros viram float: 9**9**9 passa a ser OverflowError imediato
        # (inválida), em vez de um inteiro exato de 370 milhões de dígitos.
        code = _INT.sub(lambda m: m.group(0) + ".0", code)
        counter = iter(range(self.n_const))
        code = _CONST.sub(lambda _: f"c[{next(counter)}]", code)
        code = _VAR.sub(lambda m: f"x[{int(m.group(1)) - 1}]", code)
        try:
            self._code = compile(code, "<expr>", "eval")
        except Exception as exc:
            raise InvalidExpression(f"código inválido: {exc}") from exc

    def evaluate(self, X: np.ndarray, constants: Optional[Sequence[float]] = None) -> np.ndarray:
        """Predição por linha de ``X``; ``NaN`` em tudo se a avaliação falhar."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        if self.max_var > X.shape[1]:
            return np.full(n, np.nan)
        c = np.ones(self.n_const) if constants is None else np.asarray(constants, dtype=float)
        env = {"x": [X[:, i] for i in range(X.shape[1])], "c": c, **_NP_FUNCS}
        try:
            with np.errstate(all="ignore"):
                out = eval(self._code, {"__builtins__": {}}, env)
            if np.iscomplexobj(out):
                return np.full(n, np.nan)
            return np.broadcast_to(np.asarray(out, dtype=float), (n,)).copy()
        except Exception:
            return np.full(n, np.nan)

    def fit_constants(self, X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """Ajusta as constantes por L-BFGS-B a partir de 1. Devolve o vetor ajustado."""
        ones = np.ones(self.n_const)
        if self.n_const == 0:
            return ones
        from scipy.optimize import minimize

        def loss(c):
            pred = self.evaluate(X, c)
            return float(np.mean((y - pred) ** 2)) if np.all(np.isfinite(pred)) else 1e300

        base = loss(ones)
        try:
            result = minimize(loss, ones, method="L-BFGS-B", options={"maxiter": max_iter})
        except Exception:
            return ones
        return result.x if np.all(np.isfinite(result.x)) and result.fun < base else ones


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    """R² clássico; ``-inf`` se a predição não for finita."""
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return float("-inf")
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def reference_values(text: str, X: np.ndarray, constants: Optional[Sequence[float]] = None,
                     is_prefix: bool = False) -> np.ndarray:
    """Avaliador de referência: sympy (com simplificação) -> ``math``, ponto a ponto.

    Caminho de código independente de ``Expression.evaluate``. Lento: use em poucas
    linhas. Pontos em que o cálculo falha voltam como ``NaN``.
    """
    source = prefix_to_infix(text.replace("^", "**").split()) if is_prefix else text
    _check_guards(source)
    raw = sympy.sympify(source, evaluate=False)
    n_const = len(_CONST.findall(str(raw)))
    c = [1.0] * n_const if constants is None else [float(v) for v in constants]
    counter = iter(range(n_const))
    numbered = _CONST.sub(lambda _: f"c_{next(counter)}", str(raw))
    expr = sympy.sympify(numbered)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    symbols = [sympy.Symbol(f"x_{i + 1}") for i in range(X.shape[1])]
    consts = [sympy.Symbol(f"c_{i}") for i in range(n_const)]
    fn = sympy.lambdify(symbols + consts, expr, modules=[_MATH_FUNCS, "math"])
    out = np.full(X.shape[0], np.nan)
    for row in range(X.shape[0]):
        try:
            value = fn(*X[row].tolist(), *c)
            if isinstance(value, complex):
                continue
            out[row] = float(value)
        except Exception:
            continue
    return out


def evaluators_agree(text: str, X: np.ndarray, constants: Optional[Sequence[float]] = None,
                     rtol: float = 1e-6, max_rows: int = 16, is_prefix: bool = False) -> bool:
    """``True`` se os dois avaliadores coincidem onde ambos são finitos.

    Diferença de domínio (um finito, outro ``NaN``) não conta: o sympy simplifica
    ``sqrt(x)**2`` para ``x``. Diferença de *valor* é sempre erro.
    """
    rows = np.asarray(X, dtype=float)[:max_rows]
    try:
        fast = Expression(text, is_prefix).evaluate(rows, constants)
        ref = reference_values(text, rows, constants, is_prefix=is_prefix)
    except Exception:
        return True  # inválida nos dois caminhos: nada a comparar
    both = np.isfinite(fast) & np.isfinite(ref)
    if not both.any():
        return True
    return bool(np.allclose(fast[both], ref[both], rtol=rtol, atol=1e-9))
