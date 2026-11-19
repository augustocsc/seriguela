"""
Out-of-Distribution (OOD) evaluation utilities.

Provides tools for testing model robustness:
1. Domain OOD: Different input ranges than training
2. Structural OOD: Different expression structures than training

Domain extrapolation:
- In-domain (ID): Same range as training (e.g., [0, 2])
- Near-OOD: Slightly extended range (e.g., [-0.5, 2.5])
- Far-OOD: Significantly different range (e.g., [3, 5])
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Types of domain for evaluation."""
    IN_DOMAIN = "in_domain"
    NEAR_OOD = "near_ood"
    FAR_OOD = "far_ood"
    INTERPOLATION = "interpolation"  # Denser points in training range


@dataclass
class DomainConfig:
    """Configuration for domain-based OOD evaluation."""
    domain_type: DomainType
    domain_range: Tuple[float, float]
    n_samples: int = 100


class DomainOODGenerator:
    """
    Generates out-of-distribution test data by varying input domains.

    For a training domain of [a, b], generates:
    - In-domain: [a, b] (same as training)
    - Near-OOD: [a-delta, b+delta] where delta = 0.25*(b-a)
    - Far-OOD: [b, b+range] or [-range+a, a]
    - Interpolation: Denser grid in [a, b]
    """

    def __init__(
        self,
        training_domain: Tuple[float, float],
        n_samples: int = 100,
    ):
        """
        Initialize OOD generator.

        Args:
            training_domain: Tuple of (min, max) for training data
            n_samples: Number of samples to generate
        """
        self.training_min, self.training_max = training_domain
        self.training_range = self.training_max - self.training_min
        self.n_samples = n_samples

    def get_domain_configs(self) -> Dict[str, DomainConfig]:
        """
        Get all domain configurations for evaluation.

        Returns:
            Dictionary mapping domain name to configuration
        """
        delta = 0.25 * self.training_range

        configs = {
            "in_domain": DomainConfig(
                domain_type=DomainType.IN_DOMAIN,
                domain_range=(self.training_min, self.training_max),
                n_samples=self.n_samples,
            ),
            "near_ood": DomainConfig(
                domain_type=DomainType.NEAR_OOD,
                domain_range=(
                    self.training_min - delta,
                    self.training_max + delta,
                ),
                n_samples=self.n_samples,
            ),
            "far_ood_right": DomainConfig(
                domain_type=DomainType.FAR_OOD,
                domain_range=(
                    self.training_max,
                    self.training_max + self.training_range,
                ),
                n_samples=self.n_samples,
            ),
            "far_ood_left": DomainConfig(
                domain_type=DomainType.FAR_OOD,
                domain_range=(
                    self.training_min - self.training_range,
                    self.training_min,
                ),
                n_samples=self.n_samples,
            ),
            "interpolation": DomainConfig(
                domain_type=DomainType.INTERPOLATION,
                domain_range=(self.training_min, self.training_max),
                n_samples=self.n_samples * 3,  # Denser grid
            ),
        }

        return configs

    def generate_domain_data(
        self,
        n_vars: int,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate X data for all domain configurations.

        Args:
            n_vars: Number of variables
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping domain name to X data
        """
        if seed is not None:
            np.random.seed(seed)

        configs = self.get_domain_configs()
        domain_data = {}

        for name, config in configs.items():
            x = np.random.uniform(
                config.domain_range[0],
                config.domain_range[1],
                (config.n_samples, n_vars)
            )
            domain_data[name] = x
            logger.info(
                f"Domain '{name}': range={config.domain_range}, "
                f"n_samples={config.n_samples}"
            )

        return domain_data


# Structural OOD: Different expression structures
STRUCTURAL_OOD_PROBLEMS = {
    # Problems with different structures than Nguyen benchmarks
    "polynomial_simple": {
        "equation": "x**2 + x",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
        "structure": "polynomial",
    },
    "polynomial_high_degree": {
        "equation": "x**7 + x**5 + x**3 + x",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
        "structure": "polynomial",
    },
    "rational": {
        "equation": "(x**2 + 1) / (x + 1)",
        "vars": ["x_1"],
        "domain": (0.1, 2),  # Avoid division issues
        "n_samples": 100,
        "structure": "rational",
    },
    "nested_trig": {
        "equation": "sin(cos(x))",
        "vars": ["x_1"],
        "domain": (0, 6.28),
        "n_samples": 100,
        "structure": "nested_trig",
    },
    "mixed_exp_trig": {
        "equation": "exp(sin(x))",
        "vars": ["x_1"],
        "domain": (0, 3),
        "n_samples": 100,
        "structure": "mixed",
    },
    "multivar_product": {
        "equation": "x * y * (x + y)",
        "vars": ["x_1", "x_2"],
        "domain": (0, 2),
        "n_samples": 100,
        "structure": "multivariate",
    },
    "multivar_nested": {
        "equation": "sin(x * y) + cos(x + y)",
        "vars": ["x_1", "x_2"],
        "domain": (0, 3),
        "n_samples": 100,
        "structure": "multivariate_nested",
    },
}


def generate_structural_ood_data(problem_name: str) -> Tuple[np.ndarray, np.ndarray, str, set]:
    """
    Generate data for a structural OOD problem.

    Args:
        problem_name: Name of the structural OOD problem

    Returns:
        Tuple of (X, y, equation, valid_variables)
    """
    if problem_name not in STRUCTURAL_OOD_PROBLEMS:
        raise ValueError(
            f"Unknown problem: {problem_name}. "
            f"Available: {list(STRUCTURAL_OOD_PROBLEMS.keys())}"
        )

    problem = STRUCTURAL_OOD_PROBLEMS[problem_name]
    n_vars = len(problem["vars"])
    n_samples = problem["n_samples"]
    domain = problem["domain"]

    # Generate X
    x = np.random.uniform(domain[0], domain[1], (n_samples, n_vars))

    # Build evaluation context
    local_vars = {}
    for i, var_name in enumerate(problem["vars"]):
        local_vars[var_name.replace("_", "")] = x[:, i]
        local_vars["x"] = x[:, 0] if n_vars >= 1 else None
        local_vars["y"] = x[:, 1] if n_vars >= 2 else None

    # Add safe functions
    safe_funcs = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sqrt": np.sqrt,
        "log": np.log,
        "exp": np.exp,
    }
    local_vars.update(safe_funcs)

    # Evaluate
    y = eval(problem["equation"], {"__builtins__": None}, local_vars)

    return x, y, problem["equation"], set(problem["vars"])


class OODEvaluator:
    """
    Comprehensive OOD evaluator for trained models.

    Evaluates both domain OOD and structural OOD.
    """

    def __init__(
        self,
        training_domain: Tuple[float, float] = (0, 2),
        n_samples: int = 100,
    ):
        """
        Initialize OOD evaluator.

        Args:
            training_domain: Training data domain
            n_samples: Number of samples per evaluation
        """
        self.domain_generator = DomainOODGenerator(
            training_domain=training_domain,
            n_samples=n_samples,
        )

    def get_all_ood_tests(self) -> Dict[str, dict]:
        """
        Get all OOD test configurations.

        Returns:
            Dictionary of test name -> test config
        """
        tests = {}

        # Domain OOD tests
        domain_configs = self.domain_generator.get_domain_configs()
        for name, config in domain_configs.items():
            tests[f"domain_{name}"] = {
                "type": "domain",
                "config": config,
            }

        # Structural OOD tests
        for problem_name in STRUCTURAL_OOD_PROBLEMS:
            tests[f"structural_{problem_name}"] = {
                "type": "structural",
                "problem": problem_name,
            }

        return tests


# Factory function
def create_ood_evaluator(
    training_domain: Tuple[float, float] = (0, 2),
    n_samples: int = 100,
) -> OODEvaluator:
    """Create an OOD evaluator instance."""
    return OODEvaluator(
        training_domain=training_domain,
        n_samples=n_samples,
    )


# Example usage
if __name__ == "__main__":
    # Test domain OOD generator
    print("=== Domain OOD Tests ===")
    generator = DomainOODGenerator(training_domain=(0, 2), n_samples=50)
    configs = generator.get_domain_configs()

    for name, config in configs.items():
        print(f"{name}: {config.domain_range}, n={config.n_samples}")

    print("\n=== Structural OOD Tests ===")
    for problem_name in STRUCTURAL_OOD_PROBLEMS:
        x, y, eq, vars = generate_structural_ood_data(problem_name)
        print(f"{problem_name}: {eq}, X shape: {x.shape}")

    print("\n=== All OOD Tests ===")
    evaluator = create_ood_evaluator()
    all_tests = evaluator.get_all_ood_tests()
    for test_name, test_config in all_tests.items():
        print(f"  {test_name}: {test_config['type']}")
