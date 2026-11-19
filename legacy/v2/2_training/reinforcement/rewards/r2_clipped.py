"""
R² Clipped Pure reward function.

Measures curve fit quality, clipping R² to [0, 1] range
for gradient stability during RL training.
"""

import numpy as np
from typing import Optional, Set

from .base import BaseReward, RewardResult, ErrorType


class R2ClippedReward(BaseReward):
    """
    R² Clipped Pure reward function.

    Formula: R_clip = max(0, R²)

    This is the baseline reward function that purely measures
    curve fit without any complexity penalty.
    """

    def __init__(
        self,
        valid_variables: Optional[Set[str]] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        Initialize R² Clipped reward.

        Args:
            valid_variables: Set of valid variable names
            clip_min: Minimum value to clip R² to (default: 0.0)
            clip_max: Maximum value to clip R² to (default: 1.0)
        """
        super().__init__(valid_variables)
        self.clip_min = clip_min
        self.clip_max = clip_max

    @property
    def name(self) -> str:
        return "r2_clipped"

    def compute(
        self,
        expression: str,
        x: np.ndarray,
        y: np.ndarray,
        is_prefix: bool = False
    ) -> RewardResult:
        """
        Compute clipped R² reward.

        Args:
            expression: Mathematical expression string
            x: Input data array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            is_prefix: Whether expression is in prefix notation

        Returns:
            RewardResult with clipped R² as reward
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

        # Clip R² to range
        clipped_r2 = np.clip(r2, self.clip_min, self.clip_max)

        return RewardResult(
            reward=clipped_r2,
            r2=r2,
            mse=mse,
            is_valid=True,
            complexity=complexity,
            error_type=error_type,
            expression=expression,
            fitted_constants=fitted_constants
        )
