"""
Expression validation utilities.

Validates mathematical expressions using SymPy and the Expression class.
Checks for:
- Parseability (can be parsed by SymPy)
- Validity (well-formed mathematical expression)
- Constraint adherence (uses only allowed variables and operators)
"""

import re
import logging
from typing import Optional, Set
from dataclasses import dataclass, field

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of expression validation."""

    valid: bool
    parseable: bool
    expression_str: str
    sympy_expr: Optional[sympy.Expr] = None
    error: Optional[str] = None
    variables_used: Set[str] = field(default_factory=set)
    operators_used: Set[str] = field(default_factory=set)
    has_constant: bool = False
    complexity: int = 0  # Number of nodes in expression tree


class ExpressionValidator:
    """Validates mathematical expressions."""

    # Known operators and functions
    KNOWN_OPS = {"+", "-", "*", "/", "**", "^"}
    KNOWN_FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "asin", "acos", "atan"}
    UNARY_OPS = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "asin", "acos", "atan"}
    BINARY_OPS = {"+", "-", "*", "/", "**", "^"}

    # Variable pattern: x_1, x_2, etc.
    VAR_PATTERN = re.compile(r"x_\d+")

    def __init__(self):
        """Initialize the validator."""
        self._transformations = standard_transformations + (implicit_multiplication_application,)

    def validate(self, expr_str: str, is_prefix: bool = False) -> ValidationResult:
        """
        Validate an expression.

        Args:
            expr_str: Expression string to validate.
            is_prefix: Whether the expression is in prefix notation.

        Returns:
            ValidationResult with validation details.
        """
        if not expr_str or not expr_str.strip():
            return ValidationResult(
                valid=False,
                parseable=False,
                expression_str=expr_str or "",
                error="Empty expression",
            )

        expr_str = expr_str.strip()

        try:
            if is_prefix:
                sympy_expr = self._parse_prefix(expr_str)
            else:
                sympy_expr = self._parse_infix(expr_str)

            # Extract information from parsed expression
            variables = self._extract_variables(sympy_expr)
            operators = self._extract_operators(sympy_expr)
            has_constant = self._has_constant(expr_str)
            complexity = self._calculate_complexity(sympy_expr)

            return ValidationResult(
                valid=True,
                parseable=True,
                expression_str=expr_str,
                sympy_expr=sympy_expr,
                variables_used=variables,
                operators_used=operators,
                has_constant=has_constant,
                complexity=complexity,
            )

        except Exception as e:
            return ValidationResult(
                valid=False,
                parseable=False,
                expression_str=expr_str,
                error=str(e),
            )

    def _parse_infix(self, expr_str: str) -> sympy.Expr:
        """Parse infix notation expression."""
        # Replace ^ with ** for power
        expr_str = expr_str.replace("^", "**")

        # Try to parse with sympy
        try:
            return sympy.sympify(expr_str, evaluate=False)
        except Exception:
            # Try with transformations for implicit multiplication
            return parse_expr(expr_str, transformations=self._transformations, evaluate=False)

    def _parse_prefix(self, expr_str: str) -> sympy.Expr:
        """Parse prefix notation expression to SymPy."""
        tokens = expr_str.replace("^", "**").split()
        if not tokens:
            raise ValueError("Empty token list")

        stack = []

        # Process tokens in reverse order
        for token in reversed(tokens):
            if token in self.BINARY_OPS or token in self.UNARY_OPS:
                if token in self.UNARY_OPS:
                    if len(stack) < 1:
                        raise ValueError(f"Not enough operands for {token}")
                    arg = stack.pop()
                    stack.append(f"{token}({arg})")
                else:  # Binary operator
                    if len(stack) < 2:
                        raise ValueError(f"Not enough operands for {token}")
                    right = stack.pop()
                    left = stack.pop()

                    op_map = {"+": "+", "-": "-", "*": "*", "/": "/", "**": "**", "^": "**"}
                    op = op_map.get(token, token)

                    if op in ["**", "^"]:
                        stack.append(f"({left})**({right})")
                    elif op == "/":
                        stack.append(f"({left})/({right})")
                    else:
                        stack.append(f"({left}){op}({right})")
            else:
                stack.append(token)

        if len(stack) != 1:
            raise ValueError(f"Invalid prefix expression, {len(stack)} elements remaining")

        return sympy.sympify(stack[0], evaluate=False)

    def _extract_variables(self, expr: sympy.Expr) -> Set[str]:
        """Extract variables (x_1, x_2, etc.) from expression."""
        variables = set()
        for symbol in expr.free_symbols:
            name = str(symbol)
            if self.VAR_PATTERN.match(name):
                variables.add(name)
        return variables

    def _extract_operators(self, expr: sympy.Expr) -> Set[str]:
        """Extract operators and functions from expression."""
        operators = set()

        def traverse(node):
            if isinstance(node, sympy.Add):
                operators.add("+")
            elif isinstance(node, sympy.Mul):
                operators.add("*")
            elif isinstance(node, sympy.Pow):
                operators.add("**")
            elif isinstance(node, sympy.Function):
                func_name = str(type(node).__name__).lower()
                if func_name in self.KNOWN_FUNCS:
                    operators.add(func_name)

            for arg in node.args:
                traverse(arg)

        traverse(expr)
        return operators

    def _has_constant(self, expr_str: str) -> bool:
        """Check if expression contains constant 'C'."""
        # Look for standalone C (not part of cos, etc.)
        return bool(re.search(r"\bC\b", expr_str))

    def _calculate_complexity(self, expr: sympy.Expr) -> int:
        """Calculate expression complexity (number of nodes)."""
        count = 1
        for arg in expr.args:
            count += self._calculate_complexity(arg)
        return count

    def validate_constraints(
        self,
        expr_str: str,
        allowed_vars: Optional[list] = None,
        allowed_ops: Optional[list] = None,
        is_prefix: bool = False,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that expression adheres to constraints.

        Args:
            expr_str: Expression string.
            allowed_vars: List of allowed variables (e.g., ['x_1', 'x_2']).
            allowed_ops: List of allowed operators (e.g., ['+', '-', 'sin']).
            is_prefix: Whether expression is in prefix notation.

        Returns:
            Tuple of (is_valid, error_message).
        """
        result = self.validate(expr_str, is_prefix)

        if not result.valid:
            return False, result.error

        # Check variables
        if allowed_vars is not None:
            allowed_var_set = set(allowed_vars)
            invalid_vars = result.variables_used - allowed_var_set
            if invalid_vars:
                return False, f"Invalid variables: {invalid_vars}"

        # Check operators
        if allowed_ops is not None:
            allowed_op_set = set(allowed_ops)
            invalid_ops = result.operators_used - allowed_op_set
            if invalid_ops:
                return False, f"Invalid operators: {invalid_ops}"

        return True, None


def validate_expression(expr_str: str, is_prefix: bool = False) -> ValidationResult:
    """
    Convenience function to validate an expression.

    Args:
        expr_str: Expression string.
        is_prefix: Whether expression is in prefix notation.

    Returns:
        ValidationResult.
    """
    validator = ExpressionValidator()
    return validator.validate(expr_str, is_prefix)
