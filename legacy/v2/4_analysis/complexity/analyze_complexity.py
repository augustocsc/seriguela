#!/usr/bin/env python3
"""
Analyze expression complexity to understand why model fails on Nguyen-5.
"""

import json
import re
from collections import Counter

def count_operators(expr_str):
    """Count operator occurrences."""
    ops = ['+', '-', '*', '/', 'sin', 'cos', 'sqrt', 'log', 'exp', 'pow', '**']
    counts = {}
    for op in ops:
        if op in ['**', 'pow']:
            # Count power operations
            counts['pow'] = expr_str.count('**') + expr_str.count('pow(')
        else:
            counts[op] = expr_str.count(op)
    return counts

def max_nesting_depth(expr_str):
    """Estimate max nesting depth by counting parentheses."""
    max_depth = 0
    current_depth = 0
    for char in expr_str:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    return max_depth

def main():
    # Load debug expressions
    with open("debug_expressions.json", "r") as f:
        data = json.load(f)

    all_expressions = data["all_expressions"]
    valid_exprs = [e for e in all_expressions if e["is_valid"]]

    print("="*80)
    print("COMPLEXITY ANALYSIS")
    print("="*80)
    print()

    # Target function info
    print("TARGET FUNCTION: sin(x_1**2)*cos(x_1) - 1")
    print("  - Has power operation: x_1**2")
    print("  - Has nested sin and cos")
    print("  - Has multiplication of functions")
    print("  - Nesting depth: 2")
    print()

    # Analyze valid expressions
    all_ops = Counter()
    depths = []
    has_power = 0
    has_nested_trig = 0

    for expr in valid_exprs:
        expr_str = expr["expression"]

        # Count operators
        ops = count_operators(expr_str)
        for op, count in ops.items():
            all_ops[op] += count

        # Check for power
        if ops.get('pow', 0) > 0:
            has_power += 1

        # Check for nested trig
        if 'sin(sin' in expr_str or 'sin(cos' in expr_str or 'cos(sin' in expr_str or 'cos(cos' in expr_str:
            has_nested_trig += 1

        # Depth
        depth = max_nesting_depth(expr_str)
        depths.append(depth)

    print("GENERATED VALID EXPRESSIONS STATISTICS:")
    print(f"  Total valid: {len(valid_exprs)}")
    print(f"  With power operations: {has_power} ({100*has_power/len(valid_exprs):.1f}%)")
    print(f"  With nested trig functions: {has_nested_trig} ({100*has_nested_trig/len(valid_exprs):.1f}%)")
    print(f"  Average nesting depth: {sum(depths)/len(depths):.2f}")
    print(f"  Max nesting depth: {max(depths)}")
    print()

    print("OPERATOR USAGE:")
    for op, count in sorted(all_ops.items(), key=lambda x: -x[1])[:10]:
        print(f"  {op:6s}: {count:4d} times")
    print()

    # Show examples with power
    print("="*80)
    print("EXPRESSIONS WITH POWER OPERATIONS (closest to target):")
    print("="*80)
    print()
    power_exprs = [e for e in valid_exprs if 'pow' in e["expression"] or '**' in e["expression"]]
    if power_exprs:
        for i, expr in enumerate(power_exprs[:10], 1):
            print(f"{i:2d}. R2={expr['r2']:7.4f} | {expr['expression']}")
    else:
        print("  No expressions with power operations found!")
    print()

    # Show examples with nested trig
    print("="*80)
    print("EXPRESSIONS WITH NESTED TRIGONOMETRIC FUNCTIONS:")
    print("="*80)
    print()
    nested_trig = [e for e in valid_exprs if 'sin(sin' in e["expression"] or 'sin(cos' in e["expression"] or 'cos(sin' in e["expression"] or 'cos(cos' in e["expression"]]
    if nested_trig:
        for i, expr in enumerate(nested_trig[:10], 1):
            print(f"{i:2d}. R2={expr['r2']:7.4f} | {expr['expression']}")
    else:
        print("  No expressions with nested trig functions found!")

if __name__ == "__main__":
    main()
