"""
Temperature scheduling strategies for RL training.

Provides three strategies:
1. Fixed: Constant temperature throughout training
2. Linear Annealing: Linear decrease from T_max to T_min
3. Cosine Annealing: Smooth cosine decrease from T_max to T_min
"""

import math
from abc import ABC, abstractmethod
from typing import Optional


class TemperatureScheduler(ABC):
    """
    Abstract base class for temperature schedulers.

    Temperature controls exploration vs exploitation during generation:
    - Higher temperature -> more diverse/random outputs
    - Lower temperature -> more deterministic/focused outputs
    """

    @abstractmethod
    def get_temperature(self, step: int, total_steps: int) -> float:
        """
        Get temperature for current step.

        Args:
            step: Current training step
            total_steps: Total number of training steps

        Returns:
            Temperature value
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this scheduler."""
        pass

    def get_config(self) -> dict:
        """Return scheduler configuration."""
        return {"name": self.name}


class FixedTemperature(TemperatureScheduler):
    """
    Fixed temperature throughout training.

    Use this as a baseline to compare against annealing strategies.
    """

    def __init__(self, temperature: float = 0.7):
        """
        Initialize fixed temperature scheduler.

        Args:
            temperature: Constant temperature value
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"fixed_{self.temperature}"

    def get_temperature(self, step: int, total_steps: int) -> float:
        """Return constant temperature."""
        return self.temperature

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "type": "fixed",
            "temperature": self.temperature,
        }


class LinearAnnealing(TemperatureScheduler):
    """
    Linear temperature annealing.

    Formula: T(step) = T_max - (T_max - T_min) * (step / total_steps)

    Temperature decreases linearly from T_max to T_min over training.
    """

    def __init__(
        self,
        t_max: float = 1.0,
        t_min: float = 0.5,
        warmup_steps: int = 0
    ):
        """
        Initialize linear annealing scheduler.

        Args:
            t_max: Starting (maximum) temperature
            t_min: Ending (minimum) temperature
            warmup_steps: Number of steps to stay at t_max before annealing
        """
        if t_max <= 0 or t_min <= 0:
            raise ValueError("Temperatures must be positive")
        if t_min > t_max:
            raise ValueError("t_min must be <= t_max")

        self.t_max = t_max
        self.t_min = t_min
        self.warmup_steps = warmup_steps

    @property
    def name(self) -> str:
        return f"linear_{self.t_max}_{self.t_min}"

    def get_temperature(self, step: int, total_steps: int) -> float:
        """
        Compute temperature with linear annealing.

        Args:
            step: Current training step
            total_steps: Total number of training steps

        Returns:
            Temperature at current step
        """
        # Handle warmup
        if step < self.warmup_steps:
            return self.t_max

        # Adjust for warmup
        effective_step = step - self.warmup_steps
        effective_total = max(total_steps - self.warmup_steps, 1)

        # Linear interpolation
        progress = min(effective_step / effective_total, 1.0)
        temperature = self.t_max - (self.t_max - self.t_min) * progress

        return max(temperature, self.t_min)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "type": "linear",
            "t_max": self.t_max,
            "t_min": self.t_min,
            "warmup_steps": self.warmup_steps,
        }


class CosineAnnealing(TemperatureScheduler):
    """
    Cosine temperature annealing.

    Formula: T(step) = T_min + 0.5 * (T_max - T_min) * (1 + cos(π * step / total_steps))

    Provides smoother transition than linear, with slower changes at start and end.
    """

    def __init__(
        self,
        t_max: float = 1.0,
        t_min: float = 0.5,
        warmup_steps: int = 0
    ):
        """
        Initialize cosine annealing scheduler.

        Args:
            t_max: Starting (maximum) temperature
            t_min: Ending (minimum) temperature
            warmup_steps: Number of steps to stay at t_max before annealing
        """
        if t_max <= 0 or t_min <= 0:
            raise ValueError("Temperatures must be positive")
        if t_min > t_max:
            raise ValueError("t_min must be <= t_max")

        self.t_max = t_max
        self.t_min = t_min
        self.warmup_steps = warmup_steps

    @property
    def name(self) -> str:
        return f"cosine_{self.t_max}_{self.t_min}"

    def get_temperature(self, step: int, total_steps: int) -> float:
        """
        Compute temperature with cosine annealing.

        Args:
            step: Current training step
            total_steps: Total number of training steps

        Returns:
            Temperature at current step
        """
        # Handle warmup
        if step < self.warmup_steps:
            return self.t_max

        # Adjust for warmup
        effective_step = step - self.warmup_steps
        effective_total = max(total_steps - self.warmup_steps, 1)

        # Cosine annealing
        progress = min(effective_step / effective_total, 1.0)
        cosine_value = math.cos(math.pi * progress)
        temperature = self.t_min + 0.5 * (self.t_max - self.t_min) * (1 + cosine_value)

        return max(temperature, self.t_min)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "type": "cosine",
            "t_max": self.t_max,
            "t_min": self.t_min,
            "warmup_steps": self.warmup_steps,
        }


def create_temperature_scheduler(
    scheduler_type: str,
    **kwargs
) -> TemperatureScheduler:
    """
    Factory function to create temperature scheduler.

    Args:
        scheduler_type: One of "fixed_0.7", "fixed_0.9", "linear_annealing", "cosine_annealing"
        **kwargs: Additional arguments for scheduler

    Returns:
        TemperatureScheduler instance
    """
    if scheduler_type == "fixed_0.7":
        return FixedTemperature(temperature=0.7)
    elif scheduler_type == "fixed_0.9":
        return FixedTemperature(temperature=0.9)
    elif scheduler_type == "linear_annealing":
        return LinearAnnealing(
            t_max=kwargs.get("t_max", 1.0),
            t_min=kwargs.get("t_min", 0.5),
            warmup_steps=kwargs.get("warmup_steps", 0)
        )
    elif scheduler_type == "cosine_annealing":
        return CosineAnnealing(
            t_max=kwargs.get("t_max", 1.0),
            t_min=kwargs.get("t_min", 0.5),
            warmup_steps=kwargs.get("warmup_steps", 0)
        )
    elif scheduler_type.startswith("fixed_"):
        # Handle arbitrary fixed temperature like "fixed_0.8"
        try:
            temp = float(scheduler_type.split("_")[1])
            return FixedTemperature(temperature=temp)
        except (IndexError, ValueError):
            raise ValueError(f"Invalid fixed temperature format: {scheduler_type}")
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
