#!/usr/bin/env python3
"""
Simple script to show successful and failed expressions without encoding issues.
"""

import json
import sys

def main():
    # Load debug expressions
    with open("debug_expressions.json", "r") as f:
        data = json.load(f)

    all_expressions = data["all_expressions"]

    # Categorize
    valid_exprs = [e for e in all_expressions if e["is_valid"]]
    invalid_exprs = [e for e in all_expressions if not e["is_valid"]]

    # Sort valid by R2
    valid_exprs.sort(key=lambda x: x["r2"], reverse=True)

    print("="*80)
    print("EXPRESSION ANALYSIS")
    print("="*80)
    print()
    print(f"Total expressions: {len(all_expressions)}")
    print(f"Valid: {len(valid_exprs)} ({100*len(valid_exprs)/len(all_expressions):.1f}%)")
    print(f"Invalid: {len(invalid_exprs)} ({100*len(invalid_exprs)/len(all_expressions):.1f}%)")
    print()

    # Best valid expressions
    print("="*80)
    print("TOP 10 VALID EXPRESSIONS (by R2)")
    print("="*80)
    print()
    for i, expr in enumerate(valid_exprs[:10], 1):
        r2 = expr["r2"]
        expr_str = expr["expression"]
        print(f"{i:2d}. R2={r2:7.4f} | {expr_str}")
    print()

    # Worst valid expressions
    print("="*80)
    print("BOTTOM 10 VALID EXPRESSIONS (by R2)")
    print("="*80)
    print()
    for i, expr in enumerate(valid_exprs[-10:], 1):
        r2 = expr["r2"]
        expr_str = expr["expression"]
        print(f"{i:2d}. R2={r2:7.4f} | {expr_str}")
    print()

    # Invalid expressions by type
    print("="*80)
    print("INVALID EXPRESSIONS BY TYPE")
    print("="*80)
    print()

    error_types = {}
    for e in invalid_exprs:
        et = e.get("error_type", "unknown")
        if et not in error_types:
            error_types[et] = []
        error_types[et].append(e)

    for error_type, exprs in sorted(error_types.items(), key=lambda x: -len(x[1])):
        print(f"{error_type}: {len(exprs)} cases")
        print(f"  Examples:")
        for expr in exprs[:3]:
            expr_str = expr["expression"][:70]
            print(f"    - {expr_str}")
        print()

if __name__ == "__main__":
    main()
