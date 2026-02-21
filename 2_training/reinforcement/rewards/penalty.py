"""
Penalty strategies for invalid expressions.

Provides two strategies:
1. Binary: All invalid expressions receive -1.0
2. Gradient: Differentiated penalties based on error type
"""

from enum import Enum
from typing import Dict

from .base import ErrorType, RewardResult


class PenaltyStrategy(Enum):
    """Penalty strategy for invalid expressions."""
    BINARY = "binary"      # All invalid = -1.0
    GRADIENT = "gradient"  # Differentiated penalties by error type


class PenaltyHandler:
    """
    Handles penalty assignment for invalid expressions.

    Two strategies are supported:
    1. Binary: All invalid expressions receive the same penalty (-1.0)
    2. Gradient: Different penalties based on the type of error

    Gradient penalties (from experimental plan):
    - Parsing error (syntax invalid): -1.0
    - Wrong variables (uses x_2 when only x_1 exists): -0.7
    - Produces NaN/Inf (division by zero, log negative): -0.5
    - R² negative (worse than mean): -0.3
    - R² in [0, 0.5) (very weak fit): 0.0
    """

    # Default gradient penalties as specified in experimental plan
    DEFAULT_GRADIENT_PENALTIES: Dict[ErrorType, float] = {
        ErrorType.PARSING: -1.0,
        ErrorType.VARIABLES: -0.7,
        ErrorType.NAN_INF: -0.5,
        ErrorType.NEGATIVE_R2: -0.3,
        ErrorType.WEAK_R2: 0.0,
        ErrorType.NONE: 0.0,  # No penalty for valid expressions
    }

    def __init__(
        self,
        strategy: PenaltyStrategy = PenaltyStrategy.BINARY,
        binary_penalty: float = -1.0,
        gradient_penalties: Dict[ErrorType, float] = None
    ):
        """
        Initialize penalty handler.

        Args:
            strategy: Which penalty strategy to use
            binary_penalty: Penalty value for binary strategy
            gradient_penalties: Custom gradient penalties (optional)
        """
        self.strategy = strategy
        self.binary_penalty = binary_penalty
        self.gradient_penalties = gradient_penalties or self.DEFAULT_GRADIENT_PENALTIES.copy()

    def get_penalty(self, error_type: ErrorType) -> float:
        """
        Get penalty value for an error type.

        Args:
            error_type: Type of error that occurred

        Returns:
            Penalty value (negative or zero)
        """
        if error_type == ErrorType.NONE:
            return 0.0

        if self.strategy == PenaltyStrategy.BINARY:
            return self.binary_penalty

        return self.gradient_penalties.get(error_type, self.binary_penalty)

    def apply_penalty(self, result: RewardResult) -> RewardResult:
        """
        Apply penalty to a reward result if expression is invalid.

        Args:
            result: RewardResult from reward function

        Returns:
            RewardResult with penalty applied if invalid
        """
        if result.is_valid and result.error_type == ErrorType.NONE:
            return result

        # Get penalty for this error type
        penalty = self.get_penalty(result.error_type)

        # Create new result with penalty as reward
        return RewardResult(
            reward=penalty,
            r2=result.r2,
            mse=result.mse,
            is_valid=result.is_valid,
            complexity=result.complexity,
            error_type=result.error_type,
            expression=result.expression,
            fitted_constants=result.fitted_constants
        )

    def compute_with_penalty(
        self,
        reward_fn,
        expression: str,
        x,
        y,
        is_prefix: bool = False
    ) -> RewardResult:
        """
        Convenience method to compute reward and apply penalty.

        Args:
            reward_fn: Reward function to use
            expression: Expression to evaluate
            x: Input data
            y: Target values
            is_prefix: Whether expression is in prefix notation

        Returns:
            RewardResult with penalty applied if needed
        """
        result = reward_fn.compute(expression, x, y, is_prefix)
        return self.apply_penalty(result)

    def __str__(self) -> str:
        return f"PenaltyHandler(strategy={self.strategy.value})"

    def __repr__(self) -> str:
        return self.__str__()


def create_reward_with_penalty(
    reward_type: str,
    penalty_strategy: str = "binary",
    **kwargs
):
    """
    Factory function to create reward function with penalty handler.

    Args:
        reward_type: One of "r2_clipped", "length_penalized", "sr_ic"
        penalty_strategy: One of "binary", "gradient"
        **kwargs: Additional arguments for reward function

    Returns:
        Tuple of (reward_fn, penalty_handler)
    """
    from .r2_clipped import R2ClippedReward
    from .length_penalized import LengthPenalizedReward
    from .sr_ic import SRICReward

    # Create reward function
    reward_classes = {
        "r2_clipped": R2ClippedReward,
        "length_penalized": LengthPenalizedReward,
        "sr_ic": SRICReward,
    }

    if reward_type not in reward_classes:
        raise ValueError(f"Unknown reward type: {reward_type}")

    reward_fn = reward_classes[reward_type](**kwargs)

    # Create penalty handler
    strategy = PenaltyStrategy(penalty_strategy)
    penalty_handler = PenaltyHandler(strategy=strategy)

    return reward_fn, penalty_handler
