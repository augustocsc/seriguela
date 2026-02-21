"""
Reward functions for symbolic regression RL training.

This module provides three reward functions as specified in the experimental plan:
1. R² Clipped Pure - Basic R² clipped to [0, 1]
2. Length Penalized R² - R² with penalty for long expressions
3. SR-IC - Symbolic Regression Information Criterion

Also provides penalty strategies for invalid expressions:
- Binary: All invalid = -1.0
- Gradient: Differentiated penalties by error type
"""

from .base import BaseReward, RewardResult, ErrorType
from .r2_clipped import R2ClippedReward
from .length_penalized import LengthPenalizedReward
from .sr_ic import SRICReward
from .penalty import PenaltyStrategy, PenaltyHandler, create_reward_with_penalty

__all__ = [
    "BaseReward",
    "RewardResult",
    "ErrorType",
    "R2ClippedReward",
    "LengthPenalizedReward",
    "SRICReward",
    "PenaltyStrategy",
    "PenaltyHandler",
    "create_reward_with_penalty",
]
