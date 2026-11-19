"""
Prompt builders for symbolic regression experiments.

Provides different prompt strategies:
- Standard: Normal prompt with all operators
- Oracle: Hints about the true equation structure
- Distractor: Misleading hints to test robustness
"""

import json
import logging
from typing import Set, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts for experiments."""
    STANDARD = "standard"
    ORACLE = "oracle"
    DISTRACTOR = "distractor"


@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    prompt_type: PromptType = PromptType.STANDARD
    valid_variables: Set[str] = None
    operators: List[str] = None
    oracle_operators: List[str] = None  # True operators for oracle prompt
    distractor_operators: List[str] = None  # Wrong operators for distractor


# Default operators
ALL_OPERATORS = ["sin", "cos", "exp", "log", "sqrt", "+", "-", "*", "/", "**"]

# Common operator subsets for distractors
POLYNOMIAL_OPERATORS = ["+", "-", "*", "/", "**"]
TRIGONOMETRIC_OPERATORS = ["sin", "cos", "tan", "+", "-", "*"]
EXPONENTIAL_OPERATORS = ["exp", "log", "+", "-", "*", "/"]


def extract_operators_from_equation(equation: str) -> List[str]:
    """
    Extract operators used in an equation string.

    Args:
        equation: String representation of equation

    Returns:
        List of operators found in the equation
    """
    operators_found = []

    # Check unary operators
    unary_ops = ["sin", "cos", "tan", "exp", "log", "sqrt"]
    for op in unary_ops:
        if f"{op}(" in equation or f"{op} " in equation:
            operators_found.append(op)

    # Check binary operators
    binary_ops = ["+", "-", "*", "/", "**"]
    for op in binary_ops:
        if op in equation:
            # For ** vs *, check specifically
            if op == "*" and "**" in equation:
                if equation.count("*") > equation.count("**") * 2:
                    operators_found.append("*")
            elif op == "*":
                if "*" in equation and "**" not in equation:
                    operators_found.append("*")
            else:
                operators_found.append(op)

    # Ensure ** is detected
    if "**" in equation and "**" not in operators_found:
        operators_found.append("**")

    return list(set(operators_found))


def get_misleading_operators(true_operators: List[str]) -> List[str]:
    """
    Generate misleading operators that are different from true ones.

    Args:
        true_operators: The actual operators in the equation

    Returns:
        List of operators that are NOT in the true equation
    """
    # Pick operators that are NOT in the true set
    distractor_candidates = []

    if any(op in true_operators for op in ["sin", "cos", "tan"]):
        # True has trig, give them exponentials
        distractor_candidates.extend(["exp", "log"])
    else:
        # True doesn't have trig, give them trig
        distractor_candidates.extend(["sin", "cos"])

    if "**" in true_operators:
        # True has power, remove it
        distractor_candidates.extend(["+", "-", "*"])
    else:
        # True doesn't have power, add it
        distractor_candidates.append("**")

    if "sqrt" in true_operators:
        distractor_candidates.append("exp")
    else:
        distractor_candidates.append("sqrt")

    # Always include basic arithmetic
    distractor_candidates.extend(["+", "-", "*", "/"])

    # Remove any that are actually true
    distractor_ops = [op for op in distractor_candidates if op not in true_operators]

    # Ensure we have at least some operators
    if len(distractor_ops) < 3:
        distractor_ops = [op for op in ALL_OPERATORS if op not in true_operators]

    return list(set(distractor_ops))[:6]  # Limit to 6 operators


class PromptBuilder:
    """Builds prompts for different experimental conditions."""

    def __init__(
        self,
        prompt_type: PromptType = PromptType.STANDARD,
        valid_variables: Optional[Set[str]] = None,
        ground_truth: Optional[str] = None,
    ):
        """
        Initialize prompt builder.

        Args:
            prompt_type: Type of prompt (standard, oracle, distractor)
            valid_variables: Set of valid variable names
            ground_truth: Ground truth equation (needed for oracle/distractor)
        """
        self.prompt_type = prompt_type
        self.valid_variables = valid_variables or {"x_1"}
        self.ground_truth = ground_truth

        # Extract true operators if we have ground truth
        self.true_operators = []
        if ground_truth:
            self.true_operators = extract_operators_from_equation(ground_truth)

    def build_prompt(self) -> str:
        """
        Build the prompt based on prompt type.

        Returns:
            JSON prompt string
        """
        vars_list = sorted(list(self.valid_variables))

        if self.prompt_type == PromptType.STANDARD:
            operators = ALL_OPERATORS
        elif self.prompt_type == PromptType.ORACLE:
            # Give the TRUE operators (helpful hint)
            operators = self.true_operators if self.true_operators else ALL_OPERATORS
            # Add a few extra to not make it too easy
            extra_ops = ["+", "-", "*"]
            operators = list(set(operators + extra_ops))
        elif self.prompt_type == PromptType.DISTRACTOR:
            # Give WRONG operators (misleading hint)
            operators = get_misleading_operators(self.true_operators)
            # Ensure we have basic arithmetic
            operators = list(set(operators + ["+", "-", "*"]))
        else:
            operators = ALL_OPERATORS

        prompt_dict = {
            "vars": vars_list,
            "ops": operators,
            "cons": "C",
            "expr": ""
        }

        # Remove the trailing "}" and last quote to let model complete
        prompt = json.dumps(prompt_dict)[:-2]
        return prompt

    def get_config_description(self) -> str:
        """Get a description of the prompt configuration."""
        return f"PromptType={self.prompt_type.value}, vars={self.valid_variables}"


def create_prompt_builder(
    prompt_type: str,
    valid_variables: Optional[Set[str]] = None,
    ground_truth: Optional[str] = None,
) -> PromptBuilder:
    """
    Factory function to create prompt builder.

    Args:
        prompt_type: One of "standard", "oracle", "distractor"
        valid_variables: Set of valid variable names
        ground_truth: Ground truth equation

    Returns:
        PromptBuilder instance
    """
    type_map = {
        "standard": PromptType.STANDARD,
        "oracle": PromptType.ORACLE,
        "distractor": PromptType.DISTRACTOR,
    }

    if prompt_type not in type_map:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(type_map.keys())}")

    return PromptBuilder(
        prompt_type=type_map[prompt_type],
        valid_variables=valid_variables,
        ground_truth=ground_truth,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test with Nguyen-5: sin(x**2) * cos(x) - 1
    ground_truth = "sin(x**2) * cos(x) - 1"
    valid_vars = {"x_1"}

    print("Ground truth:", ground_truth)
    print("True operators:", extract_operators_from_equation(ground_truth))
    print()

    for prompt_type in ["standard", "oracle", "distractor"]:
        builder = create_prompt_builder(
            prompt_type=prompt_type,
            valid_variables=valid_vars,
            ground_truth=ground_truth,
        )
        print(f"=== {prompt_type.upper()} ===")
        print(builder.build_prompt())
        print()
