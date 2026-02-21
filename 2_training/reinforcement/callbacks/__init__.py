"""
Callbacks for RL training.

Currently implements:
- Early stopping with multiple criteria
"""

from .early_stopping import (
    StopReason,
    EarlyStoppingConfig,
    EarlyStoppingCallback,
)

__all__ = [
    "StopReason",
    "EarlyStoppingConfig",
    "EarlyStoppingCallback",
]
