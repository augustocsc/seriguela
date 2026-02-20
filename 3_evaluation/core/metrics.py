"""
Metrics calculation utilities.

Calculates quality metrics for expression generation:
- Valid rate: Percentage of valid (parseable) expressions
- Diversity rate: Percentage of unique expressions
- Constraint adherence: Percentage following prompt constraints
- Complexity statistics
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import Counter

from .validator import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Aggregated quality metrics."""

    # Core rates
    valid_rate: float = 0.0
    parseable_rate: float = 0.0
    diversity_rate: float = 0.0

    # Counts
    total_samples: int = 0
    valid_count: int = 0
    parseable_count: int = 0
    unique_count: int = 0

    # Constraint adherence
    constraint_adherence_rate: float = 0.0
    constraint_valid_count: int = 0

    # Complexity statistics
    avg_complexity: float = 0.0
    min_complexity: int = 0
    max_complexity: int = 0

    # Length statistics
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0

    # Variable/operator usage
    variable_usage: Dict[str, int] = field(default_factory=dict)
    operator_usage: Dict[str, int] = field(default_factory=dict)

    # Error analysis
    error_types: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "valid_rate": self.valid_rate,
            "parseable_rate": self.parseable_rate,
            "diversity_rate": self.diversity_rate,
            "total_samples": self.total_samples,
            "valid_count": self.valid_count,
            "parseable_count": self.parseable_count,
            "unique_count": self.unique_count,
            "constraint_adherence_rate": self.constraint_adherence_rate,
            "constraint_valid_count": self.constraint_valid_count,
            "avg_complexity": self.avg_complexity,
            "min_complexity": self.min_complexity,
            "max_complexity": self.max_complexity,
            "avg_length": self.avg_length,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "variable_usage": self.variable_usage,
            "operator_usage": self.operator_usage,
            "error_types": self.error_types,
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Quality Metrics Summary",
            f"=" * 40,
            f"Total samples: {self.total_samples}",
            f"Valid: {self.valid_count} ({self.valid_rate:.1%})",
            f"Parseable: {self.parseable_count} ({self.parseable_rate:.1%})",
            f"Unique: {self.unique_count} ({self.diversity_rate:.1%})",
            f"Constraint adherence: {self.constraint_valid_count} ({self.constraint_adherence_rate:.1%})",
            f"",
            f"Complexity: avg={self.avg_complexity:.1f}, min={self.min_complexity}, max={self.max_complexity}",
            f"Length: avg={self.avg_length:.1f}, min={self.min_length}, max={self.max_length}",
        ]

        if self.error_types:
            lines.append("")
            lines.append("Error types:")
            for error_type, count in sorted(self.error_types.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {error_type}: {count}")

        return "\n".join(lines)


@dataclass
class SampleResult:
    """Result for a single sample."""

    idx: int
    prompt: str
    output: str
    expression: Optional[str]
    validation: ValidationResult
    constraint_valid: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "idx": self.idx,
            "prompt": self.prompt,
            "output": self.output,
            "expression": self.expression,
            "valid": self.validation.valid,
            "parseable": self.validation.parseable,
            "error": self.validation.error,
            "constraint_valid": self.constraint_valid,
            "complexity": self.validation.complexity,
            "variables": list(self.validation.variables_used),
            "operators": list(self.validation.operators_used),
        }


class MetricsCalculator:
    """Calculates quality metrics from validation results."""

    def calculate(
        self,
        results: List[SampleResult],
        allowed_vars: Optional[List[str]] = None,
        allowed_ops: Optional[List[str]] = None,
    ) -> QualityMetrics:
        """
        Calculate aggregated metrics from sample results.

        Args:
            results: List of SampleResult objects.
            allowed_vars: Optional list of allowed variables for constraint checking.
            allowed_ops: Optional list of allowed operators for constraint checking.

        Returns:
            QualityMetrics object with aggregated statistics.
        """
        if not results:
            return QualityMetrics()

        total = len(results)
        valid_count = 0
        parseable_count = 0
        constraint_valid_count = 0

        unique_expressions = set()
        complexities = []
        lengths = []

        variable_counter = Counter()
        operator_counter = Counter()
        error_counter = Counter()

        for result in results:
            validation = result.validation

            if validation.parseable:
                parseable_count += 1

            if validation.valid:
                valid_count += 1
                unique_expressions.add(result.expression)
                complexities.append(validation.complexity)
                if result.expression:
                    lengths.append(len(result.expression))

                # Track variable and operator usage
                for var in validation.variables_used:
                    variable_counter[var] += 1
                for op in validation.operators_used:
                    operator_counter[op] += 1

                # Check constraint adherence
                constraint_ok = True
                if allowed_vars is not None:
                    if not validation.variables_used.issubset(set(allowed_vars)):
                        constraint_ok = False
                if allowed_ops is not None:
                    if not validation.operators_used.issubset(set(allowed_ops)):
                        constraint_ok = False

                if constraint_ok:
                    constraint_valid_count += 1

            elif validation.error:
                # Categorize error
                error_type = self._categorize_error(validation.error)
                error_counter[error_type] += 1

        unique_count = len(unique_expressions)

        # Calculate rates
        valid_rate = valid_count / total if total > 0 else 0.0
        parseable_rate = parseable_count / total if total > 0 else 0.0
        diversity_rate = unique_count / total if total > 0 else 0.0
        constraint_rate = constraint_valid_count / total if total > 0 else 0.0

        # Calculate statistics
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
        min_complexity = min(complexities) if complexities else 0
        max_complexity = max(complexities) if complexities else 0

        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        return QualityMetrics(
            valid_rate=valid_rate,
            parseable_rate=parseable_rate,
            diversity_rate=diversity_rate,
            total_samples=total,
            valid_count=valid_count,
            parseable_count=parseable_count,
            unique_count=unique_count,
            constraint_adherence_rate=constraint_rate,
            constraint_valid_count=constraint_valid_count,
            avg_complexity=avg_complexity,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
            avg_length=avg_length,
            min_length=min_length,
            max_length=max_length,
            variable_usage=dict(variable_counter),
            operator_usage=dict(operator_counter),
            error_types=dict(error_counter),
        )

    def _categorize_error(self, error: str) -> str:
        """Categorize an error message into a type."""
        error_lower = error.lower()

        if "empty" in error_lower:
            return "empty_expression"
        elif "parse" in error_lower or "syntax" in error_lower:
            return "parse_error"
        elif "operand" in error_lower:
            return "operand_error"
        elif "token" in error_lower:
            return "tokenization_error"
        elif "symbol" in error_lower:
            return "symbol_error"
        else:
            return "other_error"

    def compare(self, metrics_list: List[QualityMetrics], names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple metrics objects.

        Args:
            metrics_list: List of QualityMetrics to compare.
            names: Names for each metrics object.

        Returns:
            Dictionary with comparison data.
        """
        if len(metrics_list) != len(names):
            raise ValueError("metrics_list and names must have same length")

        comparison = {
            "names": names,
            "valid_rates": [m.valid_rate for m in metrics_list],
            "diversity_rates": [m.diversity_rate for m in metrics_list],
            "constraint_rates": [m.constraint_adherence_rate for m in metrics_list],
            "avg_complexities": [m.avg_complexity for m in metrics_list],
            "total_samples": [m.total_samples for m in metrics_list],
        }

        # Find best performer
        best_valid_idx = comparison["valid_rates"].index(max(comparison["valid_rates"]))
        best_diversity_idx = comparison["diversity_rates"].index(max(comparison["diversity_rates"]))

        comparison["best_valid"] = names[best_valid_idx]
        comparison["best_diversity"] = names[best_diversity_idx]

        return comparison


def calculate_metrics(
    results: List[SampleResult],
    allowed_vars: Optional[List[str]] = None,
    allowed_ops: Optional[List[str]] = None,
) -> QualityMetrics:
    """
    Convenience function to calculate metrics.

    Args:
        results: List of SampleResult objects.
        allowed_vars: Optional allowed variables.
        allowed_ops: Optional allowed operators.

    Returns:
        QualityMetrics object.
    """
    calculator = MetricsCalculator()
    return calculator.calculate(results, allowed_vars, allowed_ops)
