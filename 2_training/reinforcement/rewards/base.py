"""
Base interface for reward functions in symbolic regression RL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from classes.expression import Expression


class ErrorType(Enum):
    """Types of errors that can occur during expression evaluation."""
    NONE = "none"                    # No error
    PARSING = "parsing"              # Syntax/parsing error
    VARIABLES = "variables"          # Wrong variables used
    NAN_INF = "nan_inf"              # Produces NaN or Inf
    NEGATIVE_R2 = "negative_r2"      # R² < 0
    WEAK_R2 = "weak_r2"              # R² in [0, 0.5)


@dataclass
class RewardResult:
    """Result of reward computation for an expression."""
    reward: float                    # Final reward value
    r2: float                        # R² score (may be negative or NaN)
    mse: float                       # Mean squared error
    is_valid: bool                   # Whether expression is valid
    complexity: int                  # Expression complexity (token count)
    error_type: ErrorType            # Type of error if invalid
    expression: str                  # Original expression string
    fitted_constants: Optional[list] = None  # Fitted constant values


class BaseReward(ABC):
    """
    Abstract base class for reward functions.

    All reward functions must implement:
    - compute(): Calculate reward for an expression given data
    - name: Property returning the reward function name
    """

    def __init__(self, valid_variables: Optional[Set[str]] = None,
                 max_complexity: int = 100):
        """
        Initialize reward function.

        Args:
            valid_variables: Set of valid variable names (e.g., {"x_1", "x_2"}).
                           If None, any x_N variable is considered valid.
            max_complexity: Limite de tokens da expressão. Acima disso a
                           expressão é tratada como inválida ANTES de qualquer
                           sympify — teto de memória (2026-06-14): em alvos
                           polinomiais/log o modelo explora expressões enormes
                           (grau alto) que, avaliadas aos milhares por step +
                           buffer, estouram a RAM do GRPO com buffer. Os alvos
                           reais têm ~5–15 tokens; 100 é folgado e ainda assim
                           limita o footprint a qualquer máquina (Colab incluso).
        """
        self.valid_variables = valid_variables
        self.max_complexity = max_complexity

    @abstractmethod
    def compute(
        self,
        expression: str,
        x: np.ndarray,
        y: np.ndarray,
        is_prefix: bool = False
    ) -> RewardResult:
        """
        Compute reward for an expression.

        Args:
            expression: Mathematical expression string
            x: Input data array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            is_prefix: Whether expression is in prefix notation

        Returns:
            RewardResult with reward value and metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this reward function."""
        pass

    def _parse_and_validate(
        self,
        expression: str,
        x: np.ndarray,
        y: np.ndarray,
        is_prefix: bool
    ) -> tuple:
        """
        Parse expression and validate it can be evaluated.

        Returns:
            Tuple of (Expression object, error_type, complexity)
            If parsing fails, Expression will be None
        """
        complexity = self._compute_complexity(expression, is_prefix)

        # Teto de complexidade (2026-06-14): expressões acima de max_complexity
        # tokens são rejeitadas ANTES do sympify. Sem isso, em alvos polinomiais
        # (nguyen_3) ou log (nguyen_7) o modelo explora expressões de grau muito
        # alto que custam memória ao serem simplificadas — avaliadas aos milhares
        # por step + os elites do buffer, estouram a RAM (causa raiz do OOM do
        # bon_grpo). É um teto, não um veredito de qualidade: nenhum alvo do
        # benchmark passa de ~15 tokens, então não descarta solução real.
        if self.max_complexity and complexity > self.max_complexity:
            return None, ErrorType.PARSING, complexity

        # Guarda anti-bomba (2026-06-12): potências com BASE NUMÉRICA e
        # expoente grande fazem o sympy materializar inteiros EXATOS gigantes
        # numa única alocação (ex.: 9**9**9 tem ~3.7e8 dígitos) — SIGKILL pelo
        # cgroup do pod (38GB), reproduzível em bon_grpo/nguyen_3. Expressão
        # que excede a memória da máquina é computacionalmente inavaliável:
        # inválida — o mesmo veredito que o hardware já aplicava, sem matar o
        # processo. Potências de variáveis (x**N) avaliam em float e são
        # inofensivas; só base numérica explode.
        import math as _math
        import re as _re
        for _b, _e in _re.findall(
                r"(?<![\w_.])(\d+(?:\.\d+)?)\s*\*\*\s*\(?\s*(\d+(?:\.\d+)?)", expression):
            _bf, _ef = float(_b), float(_e)
            if _bf > 1.0 and _ef * _math.log10(_bf) > 1e6:  # >1M dígitos
                return None, ErrorType.PARSING, complexity
        for _b, _e1, _e2 in _re.findall(
                r"(?<![\w_.])(\d+(?:\.\d+)?)\s*\)?\s*\*\*\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?\s*\*\*\s*\(?\s*(\d+(?:\.\d+)?)",
                expression):  # torre n**n**n (assoc. à direita): estima dígitos com caps
            _bf, _e1f, _e2f = float(_b), float(_e1), float(_e2)
            if _bf > 1.0 and _e1f > 1.0:
                _exp_log = _e2f * _math.log10(_e1f)        # log10 do expoente efetivo
                if _exp_log > 12 or (10 ** min(_exp_log, 12)) * _math.log10(_bf) > 1e6:
                    return None, ErrorType.PARSING, complexity

        # Guarda anti-HANG (2026-06-16): potência de QUALQUER base (inclusive
        # variável) com expoente numérico grande — ex.: x_1**(9**9) = x_1**387420489 —
        # NÃO estoura memória (base é variável, avalia em float), mas faz o numpy
        # entrar em quadrados sucessivos (um.multiply(x,x,out=x)) e TRAVAR por horas.
        # Observado em bon_grpo/nguyen_3 no step 49 (GPU 0%, log congelado). Corrige
        # a suposição errada do guard anterior ("x**N é inofensivo"). Alvos reais têm
        # expoentes ≤ ~6; teto de 100 é folgado. Dois casos:
        _MAX_EXP = 100
        for _e in _re.findall(r"\*\*\s*\(?\s*-?(\d+(?:\.\d+)?)", expression):
            if float(_e) > _MAX_EXP:                       # expoente literal grande
                return None, ErrorType.PARSING, complexity
        # expoente que é ele próprio uma potência — dois casos, sem atravessar
        # operadores (o alvo x**5 + x**4 NÃO pode casar):
        if _re.search(r"\*\*\s*\([^)]*\*\*", expression) or \
           _re.search(r"\*\*\s*\d+(?:\.\d+)?\s*\*\*", expression):  # x**(9**9) / x**2**3
            return None, ErrorType.PARSING, complexity

        # Try to parse
        try:
            expr = Expression(expression, is_prefix=is_prefix)
        except Exception as e:
            return None, ErrorType.PARSING, complexity

        # Check variables if specified
        if self.valid_variables is not None:
            expr_vars = self._extract_variables(expression)
            invalid_vars = expr_vars - self.valid_variables
            if invalid_vars:
                return None, ErrorType.VARIABLES, complexity

        # Check if expression is valid on dataset
        if not expr.is_valid_on_dataset(x):
            return None, ErrorType.NAN_INF, complexity

        return expr, ErrorType.NONE, complexity

    def _compute_complexity(self, expression: str, is_prefix: bool) -> int:
        """
        Compute expression complexity as token count.

        For prefix notation, split by whitespace.
        For infix notation, count operators and operands.
        """
        if is_prefix:
            return len(expression.split())
        else:
            # Count tokens in infix (rough approximation)
            import re
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[+\-*/^()]', expression)
            return len(tokens)

    def _extract_variables(self, expression: str) -> Set[str]:
        """Extract variable names (x_1, x_2, etc.) from expression."""
        import re
        return set(re.findall(r'x_\d+', expression))

    def _compute_r2_and_mse(
        self,
        expr: Expression,
        x: np.ndarray,
        y: np.ndarray
    ) -> tuple:
        """
        Fit constants and compute R² and MSE.

        Returns:
            Tuple of (r2, mse, fitted_constants)
        """
        try:
            r2 = expr.fit_constants(x, y)
            y_pred = expr.evaluate(x)

            if not np.all(np.isfinite(y_pred)):
                return -np.inf, np.inf, expr.best_constants

            mse = np.mean((y - y_pred) ** 2)
            return r2, mse, expr.best_constants

        except Exception:
            return -np.inf, np.inf, []

    def _classify_r2(self, r2: float) -> ErrorType:
        """Classify R² value into error type."""
        if r2 < 0:
            return ErrorType.NEGATIVE_R2
        elif r2 < 0.5:
            return ErrorType.WEAK_R2
        else:
            return ErrorType.NONE
