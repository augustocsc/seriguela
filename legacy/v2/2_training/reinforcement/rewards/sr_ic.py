"""
Symbolic Regression Information Criterion (SR-IC) reward function.

Balances MSE with expression complexity using a logarithmic criterion.
"""

import numpy as np
from typing import Optional, Set

from .base import BaseReward, RewardResult, ErrorType


class SRICReward(BaseReward):
    """
    Symbolic Regression Information Criterion reward function.

    Formula: R_SRIC = -log(MSE + ε) - λ * C

    Where:
        - MSE is the mean squared error
        - ε is a small constant to prevent log(0) (default: 1e-10)
        - λ is the complexity weight (default: 0.1)
        - C is the expression complexity (token count)

    The negative sign transforms minimization into maximization for RL.
    This criterion naturally balances fit quality and complexity.
    """

    def __init__(
        self,
        lambda_complexity: float = 0.1,
        epsilon: float = 1e-10,
        valid_variables: Optional[Set[str]] = None,
        normalize: bool = True
    ):
        """
        Initialize SR-IC reward.

        Args:
            lambda_complexity: Weight for complexity penalty
            epsilon: Small constant to prevent log(0)
            valid_variables: Set of valid variable names
            normalize: Whether to normalize reward to roughly [0, 1] range
        """
        super().__init__(valid_variables)
        self.lambda_c = lambda_complexity
        self.epsilon = epsilon
        self.normalize = normalize
        # For normalization: -log(1e-10) ≈ 23, typical good MSE ~0.01 -> -log(0.01) ≈ 4.6
        self.max_expected_reward = 25.0  # Approximate upper bound

    @property
    def name(self) -> str:
        return f"sr_ic_lambda{self.lambda_c}"

    def compute(
        self,
        expression: str,
        x: np.ndarray,
        y: np.ndarray,
        is_prefix: bool = False
    ) -> RewardResult:
        """
        Compute SR-IC reward.

        Args:
            expression: Mathematical expression string
            x: Input data array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            is_prefix: Whether expression is in prefix notation

        Returns:
            RewardResult with SR-IC reward
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
        if not np.isfinite(r2) or not np.isfinite(mse):
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

        # Compute SR-IC reward
        # R_SRIC = -log(MSE + ε) - λ * C
        log_term = -np.log(mse + self.epsilon)
        complexity_penalty = self.lambda_c * complexity
        reward = log_term - complexity_penalty

        # Optionally normalize to roughly [0, 1] range
        if self.normalize:
            # Shift and scale to make typical rewards in [0, 1]
            reward = max(0.0, reward / self.max_expected_reward)

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
