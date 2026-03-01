#!/usr/bin/env python3
"""
Analyze Pre-Phase B Validation Results

Reads all test summary JSONs from results/pre_phase_b/ and produces
a consolidated go/no-go report for Phase B.

Usage:
    python scripts/pre_phase_b/analyze_results.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results/pre_phase_b")

# ─── Analysis Functions ───────────────────────────────────────────────────────

def analyze_test1():
    """Test 1: Multi-Seed Robustness."""
    summary_path = RESULTS_DIR / "test1_multi_seed" / "test1_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        data = json.load(f)

    results = data["results"]
    problems = data["problems"]
    seeds = data["seeds"]

    print("\n" + "="*70)
    print("  TEST 1: MULTI-SEED ROBUSTNESS")
    print("="*70)

    all_ok = True
    for problem in problems:
        r2_values = [r["best_r2"] for r in results
                     if r["problem"] == problem and r["best_r2"] > -900]
        if not r2_values:
            print(f"  {problem}: NO RESULTS")
            all_ok = False
            continue

        mean_r2 = sum(r2_values) / len(r2_values)
        std_r2 = (sum((v - mean_r2)**2 for v in r2_values) / len(r2_values)) ** 0.5
        max_r2 = max(r2_values)
        min_r2 = min(r2_values)

        status = "✅" if std_r2 < 0.1 else "⚠️"
        if std_r2 >= 0.1:
            all_ok = False

        print(f"  {problem:<12} mean={mean_r2:.4f}  σ={std_r2:.4f}  "
              f"range=[{min_r2:.4f}, {max_r2:.4f}]  {status}")

    verdict = "PASS" if all_ok else "NEEDS ATTENTION"
    print(f"\n  Verdict: {verdict}")
    print(f"  Criterion: σ(R²) < 0.1 for all problems")
    return all_ok


def analyze_test2():
    """Test 2: Nguyen-5 Failure Debug."""
    summary_path = RESULTS_DIR / "test2_nguyen5_debug" / "test2_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        data = json.load(f)

    results = data["results"]
    temperatures = data["temperatures"]

    print("\n" + "="*70)
    print("  TEST 2: NGUYEN-5 FAILURE DEBUG")
    print("="*70)

    any_success = False
    for temp in temperatures:
        temp_results = [r for r in results
                        if r["temperature"] == temp and r.get("best_r2", -999) > -900]
        if not temp_results:
            print(f"  {temp}: NO RESULTS")
            continue

        r2_values = [r["best_r2"] for r in temp_results]
        mean_r2 = sum(r2_values) / len(r2_values)
        max_r2 = max(r2_values)
        success = any(v >= 0.5 for v in r2_values)
        if success:
            any_success = True

        status = "✅" if success else "❌"
        print(f"  {temp:<25} mean={mean_r2:.4f}  max={max_r2:.4f}  "
              f"any≥0.5={'YES' if success else 'NO'}  {status}")

        for r in temp_results:
            expr = r.get("best_expression", "N/A")
            print(f"    seed={r['seed']}: R²={r['best_r2']:.4f}  expr={expr}")

    verdict = "SOLVABLE" if any_success else "INHERENTLY HARD (expected)"
    print(f"\n  Verdict: {verdict}")
    print(f"  Note: Nguyen-5 is known to be difficult. Failure here is not a blocker.")
    return any_success


def analyze_test3():
    """Test 3: Convergence Profile (200 Steps)."""
    summary_path = RESULTS_DIR / "test3_convergence" / "test3_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        data = json.load(f)

    results = data["results"]

    print("\n" + "="*70)
    print("  TEST 3: CONVERGENCE PROFILE (200 STEPS)")
    print("="*70)

    recommendations = {}
    for r in results:
        problem = r["problem"]
        trajectory = r.get("r2_trajectory", [])
        best_r2 = r.get("best_r2", -999)
        total_steps = r.get("total_steps", 0)

        if not trajectory:
            print(f"  {problem}: NO TRAJECTORY DATA")
            continue

        # Analyze trajectory shape
        first_r2 = trajectory[0][1] if trajectory else 0
        last_r2 = trajectory[-1][1] if trajectory else 0

        # Check if R² improved in last 50 steps
        if len(trajectory) > 50:
            r2_at_150 = trajectory[-51][1] if len(trajectory) > 50 else first_r2
            improvement_last_50 = last_r2 - r2_at_150
            still_improving = improvement_last_50 > 0.01
        else:
            improvement_last_50 = last_r2 - first_r2
            still_improving = improvement_last_50 > 0.01

        status = "📈 still improving" if still_improving else "📊 plateaued"
        print(f"  {problem:<12} best_r2={best_r2:.4f}  steps={total_steps}  "
              f"trajectory: {first_r2:.4f} → {last_r2:.4f}  {status}")

        if still_improving:
            recommendations[problem] = "increase max_steps"
        else:
            recommendations[problem] = "current patience OK"

    print(f"\n  Patience recommendation:")
    for problem, rec in recommendations.items():
        print(f"    {problem}: {rec}")

    return recommendations


def analyze_test4():
    """Test 4: Temperature Comparison."""
    summary_path = RESULTS_DIR / "test4_temp_compare" / "test4_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        data = json.load(f)

    results = data["results"]
    temperatures = data["temperatures"]

    print("\n" + "="*70)
    print("  TEST 4: TEMPERATURE COMPARISON")
    print("="*70)

    # Ranking by average R²
    temp_scores = {}
    for temp in temperatures:
        temp_results = [r for r in results
                        if r["temperature"] == temp and r.get("best_r2", -999) > -900]
        if temp_results:
            r2_values = [r["best_r2"] for r in temp_results]
            avg = sum(r2_values) / len(r2_values)
            temp_scores[temp] = avg

    ranked = sorted(temp_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (temp, avg) in enumerate(ranked, 1):
        marker = "👑" if rank == 1 else "  "
        print(f"  {marker} {rank}. {temp:<25} avg_r2={avg:.4f}")

    # Check if cosine is still #1
    cosine_wins = ranked[0][0] == "cosine_annealing" if ranked else False
    print(f"\n  Cosine annealing still #1: {'✅ YES' if cosine_wins else '⚠️ NO'}")

    return cosine_wins


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  PRE-PHASE B — CONSOLIDATED ANALYSIS")
    print("="*70)

    tests_found = 0
    tests_passed = 0

    # Test 1
    result1 = analyze_test1()
    if result1 is not None:
        tests_found += 1
        if result1:
            tests_passed += 1

    # Test 2
    result2 = analyze_test2()
    if result2 is not None:
        tests_found += 1
        # Nguyen-5 failure is expected — not a blocker
        tests_passed += 1

    # Test 3
    result3 = analyze_test3()
    if result3 is not None:
        tests_found += 1
        tests_passed += 1  # Informational test, always "passes"

    # Test 4
    result4 = analyze_test4()
    if result4 is not None:
        tests_found += 1
        if result4:
            tests_passed += 1

    # Final verdict
    print("\n" + "="*70)
    print("  FINAL VERDICT")
    print("="*70)

    if tests_found == 0:
        print("  ❌ No test results found!")
        print(f"  Expected results in: {RESULTS_DIR.absolute()}")
        print(f"  Run the test scripts first.")
        sys.exit(1)

    print(f"  Tests found: {tests_found}/4")
    print(f"  Tests passed: {tests_passed}/{tests_found}")

    if tests_found >= 3 and tests_passed >= tests_found - 1:
        print(f"\n  ✅ GO FOR PHASE B")
        print(f"  The winner config is validated. Proceed with full benchmark.")
    elif tests_found >= 2 and tests_passed >= tests_found:
        print(f"\n  ✅ PROVISIONAL GO")
        print(f"  Results look good but some tests are missing. Consider running remaining tests.")
    else:
        print(f"\n  ⚠️  REVIEW NEEDED")
        print(f"  Some validation tests failed. Review the detailed output above.")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
