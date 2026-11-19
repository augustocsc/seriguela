"""
Noise generators for robustness testing.

Provides different noise injection strategies:
- Gaussian: Standard Gaussian noise
- Uniform: Uniform random noise
- SNR-based: Signal-to-Noise Ratio based noise
"""

import numpy as np
from typing import Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Types of noise for robustness testing."""
    NONE = "none"
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    noise_type: NoiseType = NoiseType.NONE
    noise_level: float = 0.0  # Standard deviation for Gaussian, range for Uniform
    snr_db: Optional[float] = None  # Signal-to-Noise Ratio in dB


def add_gaussian_noise(
    y: np.ndarray,
    noise_level: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to target values.

    Args:
        y: Original target values
        noise_level: Standard deviation of noise (as fraction of signal std)
        seed: Random seed for reproducibility

    Returns:
        Noisy target values
    """
    if seed is not None:
        np.random.seed(seed)

    # Scale noise relative to signal
    signal_std = np.std(y)
    noise_std = noise_level * signal_std

    noise = np.random.normal(0, noise_std, size=y.shape)
    return y + noise


def add_uniform_noise(
    y: np.ndarray,
    noise_level: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add uniform noise to target values.

    Args:
        y: Original target values
        noise_level: Range of noise (as fraction of signal range)
        seed: Random seed for reproducibility

    Returns:
        Noisy target values
    """
    if seed is not None:
        np.random.seed(seed)

    signal_range = np.max(y) - np.min(y)
    noise_range = noise_level * signal_range

    noise = np.random.uniform(-noise_range/2, noise_range/2, size=y.shape)
    return y + noise


def add_snr_noise(
    y: np.ndarray,
    snr_db: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add noise based on Signal-to-Noise Ratio.

    Args:
        y: Original target values
        snr_db: Desired SNR in decibels
        seed: Random seed for reproducibility

    Returns:
        Noisy target values
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate signal power
    signal_power = np.mean(y ** 2)

    # Calculate required noise power from SNR
    # SNR_dB = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / 10^(SNR_dB / 10)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)

    noise = np.random.normal(0, noise_std, size=y.shape)
    return y + noise


def add_noise(
    y: np.ndarray,
    config: NoiseConfig,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add noise to target values based on configuration.

    Args:
        y: Original target values
        config: Noise configuration
        seed: Random seed for reproducibility

    Returns:
        Noisy target values (or original if no noise)
    """
    if config.noise_type == NoiseType.NONE:
        return y

    if config.snr_db is not None:
        return add_snr_noise(y, config.snr_db, seed)

    if config.noise_type == NoiseType.GAUSSIAN:
        return add_gaussian_noise(y, config.noise_level, seed)
    elif config.noise_type == NoiseType.UNIFORM:
        return add_uniform_noise(y, config.noise_level, seed)

    return y


class NoiseGenerator:
    """
    Generates noisy versions of data for robustness testing.

    Supports multiple noise levels for systematic evaluation.
    """

    # Standard noise levels for experiments
    STANDARD_NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]

    def __init__(
        self,
        noise_type: str = "gaussian",
        noise_levels: Optional[list] = None,
    ):
        """
        Initialize noise generator.

        Args:
            noise_type: Type of noise ("none", "gaussian", "uniform")
            noise_levels: List of noise levels to test
        """
        self.noise_type = NoiseType(noise_type)
        self.noise_levels = noise_levels or self.STANDARD_NOISE_LEVELS

    def generate_noisy_data(
        self,
        y: np.ndarray,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Generate multiple noisy versions of data.

        Args:
            y: Original target values
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping noise_level -> noisy_y
        """
        noisy_data = {}

        for level in self.noise_levels:
            config = NoiseConfig(
                noise_type=self.noise_type,
                noise_level=level,
            )
            noisy_y = add_noise(y, config, seed)
            noisy_data[level] = noisy_y

            if level > 0:
                # Calculate actual SNR
                signal_power = np.mean(y ** 2)
                noise = noisy_y - y
                noise_power = np.mean(noise ** 2)
                if noise_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                    logger.info(f"Noise level {level}: SNR = {snr_db:.2f} dB")

        return noisy_data


def create_noise_config(
    noise_type: str = "none",
    noise_level: float = 0.0,
    snr_db: Optional[float] = None,
) -> NoiseConfig:
    """
    Factory function to create noise configuration.

    Args:
        noise_type: Type of noise ("none", "gaussian", "uniform")
        noise_level: Noise level (fraction of signal)
        snr_db: Optional SNR in dB (overrides noise_level)

    Returns:
        NoiseConfig instance
    """
    return NoiseConfig(
        noise_type=NoiseType(noise_type),
        noise_level=noise_level,
        snr_db=snr_db,
    )


# Example usage
if __name__ == "__main__":
    # Test noise generation
    np.random.seed(42)

    # Generate sample data
    x = np.linspace(0, 2, 100)
    y = np.sin(x ** 2) * np.cos(x) - 1  # Nguyen-5

    print("Original signal:")
    print(f"  Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}")
    print()

    # Test different noise levels
    generator = NoiseGenerator(noise_type="gaussian")
    noisy_data = generator.generate_noisy_data(y, seed=42)

    for level, noisy_y in noisy_data.items():
        print(f"Noise level {level}:")
        print(f"  Mean: {np.mean(noisy_y):.4f}, Std: {np.std(noisy_y):.4f}")
        if level > 0:
            rmse = np.sqrt(np.mean((y - noisy_y) ** 2))
            print(f"  RMSE: {rmse:.4f}")
        print()
