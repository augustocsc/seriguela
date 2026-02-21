"""
Length-Penalized R² reward function.

Penalizes long expressions to encourage parsimonious solutions.
"""

import numpy as np
from typing import Optional, Set

from .base import BaseReward, RewardResult, ErrorType


class LengthPenalizedReward(BaseReward):
    """
    R² with length penalty reward function.

    Formula: R_length = R² - α * L

    Where:
        - R² is the coefficient of determination
        - α is the penalty coefficient (default: 0.01)
        - L is the number of tokens in the expression

    This encourages shorter expressions with equivalent fit.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        valid_variables: Optional[Set[str]] = None,
        clip_r2: bool = True
    ):
        """
        Initialize Length-Penalized reward.

        Args:
            alpha: Penalty coefficient for expression length
            valid_variables: Set of valid variable names
            clip_r2: Whether to clip R² to [0, 1] before penalty
        """
        super().__init__(valid_variables)
        self.alpha = alpha
        self.clip_r2 = clip_r2

    @property
    def name(self) -> str:
        return f"length_penalized_alpha{self.alpha}"

    def compute(
        self,
        expression: str,
        x: np.ndarray,
        y: np.ndarray,
        is_prefix: bool = False
    ) -> RewardResult:
        """
        Compute length-penalized R² reward.

        Args:
            expression: Mathematical expression string
            x: Input data array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            is_prefix: Whether expression is in prefix notation

        Returns:
            RewardResult with length-penalized reward
        """
        # Parse and validate
        expr, error_type, complexity = self._parse_and_validate(
            expression, x, y, is_prefix
        )

        # If parsing failed, return invalid result
        if expr is None:
            return RewardResult(
                reward=0.0,  # Will be replaced by penalty handler
                r2=-np.inf,
                mse=np.inf,
                is_valid=False,
                complexity=complexity,
                error_type=error_type,
                expression=expression,
                fitted_constants=None
            )

        # Compute R² and MSE
        r2, mse, fitted_constants = self._compute_r2_and_mse(expr, x, y)

        # Check if computation produced valid results
        if not np.isfinite(r2):
            return RewardResult(
                reward=0.0,
                r2=r2,
                mse=mse,
                is_valid=False,
                complexity=complexity,
                error_type=ErrorType.NAN_INF,
                expression=expression,
                fitted_constants=fitted_constants
            )

        # Classify R² for error type
        error_type = self._classify_r2(r2)

        # Compute penalized reward
        base_r2 = np.clip(r2, 0.0, 1.0) if self.clip_r2 else r2
        length_penalty = self.alpha * complexity
        reward = base_r2 - length_penalty

        return RewardResult(
            reward=reward,
            r2=r2,
            mse=mse,
            is_valid=True,
            complexity=complexity,
            error_type=error_type,
            expression=expression,
            fitted_constants=fitted_constants
        )
