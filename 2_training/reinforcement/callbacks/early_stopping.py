"""
Early stopping callbacks for RL training.

Implements four stopping criteria as specified in the experimental plan:
1. Convergence: Reward not improving for N epochs
2. Exact Recovery: R² > 0.999 and expression matches ground truth
3. Policy Collapse: Entropy below threshold
4. Max Steps: Hard limit on training steps
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Callable
import numpy as np


class StopReason(Enum):
    """Reasons for stopping training."""
    NONE = "none"                      # Continue training
    CONVERGENCE = "convergence"         # Reward converged (no improvement)
    EXACT_RECOVERY = "exact_recovery"   # Found exact solution
    POLICY_COLLAPSE = "policy_collapse" # Policy entropy too low
    MAX_STEPS = "max_steps"             # Reached maximum steps
    USER_INTERRUPT = "user_interrupt"   # User requested stop


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for early stopping criteria.

    All four criteria from experimental plan are configurable:
    1. Convergence: patience epochs without δ improvement
    2. Exact Recovery: R² >= threshold with symbolic match
    3. Policy Collapse: entropy < threshold
    4. Max Steps: step > max_steps
    """
    # Convergence criterion
    patience: int = 5
    delta: float = 0.01
    use_convergence: bool = True

    # Exact recovery criterion
    r2_threshold: float = 0.999
    check_symbolic_match: bool = True
    use_exact_recovery: bool = True

    # Policy collapse criterion
    entropy_threshold: float = 0.1
    use_policy_collapse: bool = True

    # Max steps criterion
    max_steps: int = 10000
    use_max_steps: bool = True


@dataclass
class EarlyStoppingState:
    """Internal state for early stopping tracking."""
    reward_history: List[float] = field(default_factory=list)
    best_reward: float = -float('inf')
    best_r2: float = -float('inf')
    best_expression: Optional[str] = None
    steps_without_improvement: int = 0
    total_steps: int = 0


class EarlyStoppingCallback:
    """
    Manages multiple early stopping criteria.

    Usage:
        callback = EarlyStoppingCallback(config, ground_truth="x**2 + 1")

        for step in range(max_steps):
            # ... training step ...
            stop_reason = callback.check(
                step=step,
                mean_reward=mean_reward,
                best_r2=best_r2,
                best_expr=best_expression,
                policy_entropy=entropy
            )
            if stop_reason != StopReason.NONE:
                print(f"Stopping: {stop_reason.value}")
                break
    """

    def __init__(
        self,
        config: EarlyStoppingConfig,
        ground_truth: Optional[str] = None,
        symbolic_checker: Optional[Callable[[str, str], bool]] = None
    ):
        """
        Initialize early stopping callback.

        Args:
            config: Early stopping configuration
            ground_truth: Ground truth expression (for exact recovery check)
            symbolic_checker: Function to check symbolic equivalence.
                             Signature: (generated_expr, ground_truth) -> bool
        """
        self.config = config
        self.ground_truth = ground_truth
        self.symbolic_checker = symbolic_checker
        self.state = EarlyStoppingState()

    def reset(self):
        """Reset state for new training run."""
        self.state = EarlyStoppingState()

    def state_dict(self) -> dict:
        """Get callback state for checkpointing."""
        return {
            "reward_history": self.state.reward_history,
            "best_reward": self.state.best_reward,
            "best_r2": self.state.best_r2,
            "best_expression": self.state.best_expression,
            "steps_without_improvement": self.state.steps_without_improvement,
            "total_steps": self.state.total_steps,
        }

    def load_state_dict(self, state: dict):
        """Load callback state from checkpoint."""
        self.state.reward_history = state.get("reward_history", [])
        self.state.best_reward = state.get("best_reward", -float('inf'))
        self.state.best_r2 = state.get("best_r2", -float('inf'))
        self.state.best_expression = state.get("best_expression")
        self.state.steps_without_improvement = state.get("steps_without_improvement", 0)
        self.state.total_steps = state.get("total_steps", 0)

    def check(
        self,
        step: int,
        mean_reward: float,
        best_r2: float,
        best_expr: str,
        policy_entropy: float
    ) -> StopReason:
        """
        Check all stopping criteria.

        Args:
            step: Current training step
            mean_reward: Mean reward of current batch
            best_r2: Best R² score found so far
            best_expr: Best expression found so far
            policy_entropy: Current policy entropy

        Returns:
            StopReason indicating why to stop, or NONE to continue
        """
        self.state.total_steps = step

        # Update best tracking
        if best_r2 > self.state.best_r2:
            self.state.best_r2 = best_r2
            self.state.best_expression = best_expr

        # Check criteria in order of priority

        # 1. Max steps (highest priority - hard limit)
        if self.config.use_max_steps and step >= self.config.max_steps:
            return StopReason.MAX_STEPS

        # 2. Exact recovery (success condition)
        if self.config.use_exact_recovery:
            if best_r2 >= self.config.r2_threshold:
                if self._check_symbolic_match(best_expr):
                    return StopReason.EXACT_RECOVERY

        # 3. Policy collapse (failure condition)
        if self.config.use_policy_collapse:
            if policy_entropy < self.config.entropy_threshold:
                return StopReason.POLICY_COLLAPSE

        # 4. Convergence (no improvement)
        if self.config.use_convergence:
            self.state.reward_history.append(mean_reward)

            if mean_reward > self.state.best_reward + self.config.delta:
                self.state.best_reward = mean_reward
                self.state.steps_without_improvement = 0
            else:
                self.state.steps_without_improvement += 1

            if self.state.steps_without_improvement >= self.config.patience:
                return StopReason.CONVERGENCE

        return StopReason.NONE

    def _check_symbolic_match(self, expression: str) -> bool:
        """
        Check if expression matches ground truth symbolically.

        Args:
            expression: Generated expression

        Returns:
            True if matches (or if check is disabled)
        """
        if not self.config.check_symbolic_match:
            return True

        if self.ground_truth is None:
            return True

        # Use provided checker if available
        if self.symbolic_checker is not None:
            try:
                return self.symbolic_checker(expression, self.ground_truth)
            except Exception:
                return False

        # Default: use SymPy for symbolic comparison
        return self._sympy_equivalence(expression, self.ground_truth)

    def _sympy_equivalence(self, expr1: str, expr2: str) -> bool:
        """
        Check symbolic equivalence using SymPy.

        Args:
            expr1: First expression
            expr2: Second expression

        Returns:
            True if symbolically equivalent
        """
        try:
            import sympy
            from sympy import simplify, expand, sympify

            # Parse expressions
            e1 = sympify(expr1)
            e2 = sympify(expr2)

            # Check if difference simplifies to zero
            diff = simplify(expand(e1 - e2))
            return diff == 0

        except Exception:
            # If SymPy fails, fall back to string comparison
            return expr1.strip() == expr2.strip()

    def get_summary(self) -> dict:
        """Get summary of early stopping state."""
        return {
            "total_steps": self.state.total_steps,
            "best_reward": self.state.best_reward,
            "best_r2": self.state.best_r2,
            "best_expression": self.state.best_expression,
            "steps_without_improvement": self.state.steps_without_improvement,
            "reward_history_length": len(self.state.reward_history),
        }

    def get_config_dict(self) -> dict:
        """Get configuration as dictionary."""
        return {
            "patience": self.config.patience,
            "delta": self.config.delta,
            "r2_threshold": self.config.r2_threshold,
            "check_symbolic_match": self.config.check_symbolic_match,
            "entropy_threshold": self.config.entropy_threshold,
            "max_steps": self.config.max_steps,
            "ground_truth": self.ground_truth,
        }
