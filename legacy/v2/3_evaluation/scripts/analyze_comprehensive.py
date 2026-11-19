#!/usr/bin/env python3
"""
Analyze results from comprehensive evaluation.

Generates summary tables and insights across:
- Model sizes (base, medium, large)
- Notations (infix, prefix)
- Temperature variations
- Prompt configurations
- Sampling methods
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RunResult:
    """Single run result."""
    run_id: str
    model: str
    model_size: str  # base, medium, large
    notation: str  # infix, prefix
    temperature: float
    top_p: float
    top_k: int
    vars: List[str]
    ops: List[str]
    valid_rate: float
    diversity_rate: float
    constraint_rate: float
    experiment_type: str  # temperature_sweep, prompt_variation, sampling_method


def parse_model_info(model_path: str) -> tuple:
    """Extract model size and notation from model path."""
    model_lower = model_path.lower()

    # Detect size
    if "large" in model_lower:
        size = "large"
    elif "medium" in model_lower:
        size = "medium"
    else:
        size = "base"

    # Detect notation
    notation = "prefix" if "prefix" in model_lower else "infix"

    return size, notation


def infer_experiment_type(config: dict) -> str:
    """Infer experiment type from config."""
    # Check prompt complexity
    vars_count = len(config.get("prompt", {}).get("vars", ["x_1"]))
    ops = config.get("prompt", {}).get("ops", [])

    if vars_count > 1 or any(op in ops for op in ["sin", "cos", "exp", "log"]):
        return "prompt_variation"

    temp = config.get("generation", {}).get("temperature", 0.7)
    top_p = config.get("generation", {}).get("top_p", 0.9)

    if top_p != 0.9:
        return "sampling_method"

    return "temperature_sweep"


def load_results(results_dir: Path) -> List[RunResult]:
    """Load all results from the comprehensive evaluation."""
    results = []

    run_dirs = sorted(results_dir.glob("run_*"))

    for run_dir in run_dirs:
        config_file = run_dir / "config.yaml"
        metrics_file = run_dir / "metrics.json"

        if not config_file.exists() or not metrics_file.exists():
            continue

        # Load config
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Load metrics
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Parse model info
        model_path = config.get("model", {}).get("path", "")
        size, notation = parse_model_info(model_path)

        # Extract other config values
        gen_config = config.get("generation", {})
        prompt_config = config.get("prompt", {})

        result = RunResult(
            run_id=run_dir.name,
            model=model_path,
            model_size=size,
            notation=notation,
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            top_k=gen_config.get("top_k", 50),
            vars=prompt_config.get("vars", ["x_1"]),
            ops=prompt_config.get("ops", []),
            valid_rate=metrics.get("valid_rate", 0),
            diversity_rate=metrics.get("diversity_rate", 0),
            constraint_rate=metrics.get("constraint_adherence_rate", 0),
            experiment_type=infer_experiment_type(config),
        )
        results.append(result)

    return results


def generate_temperature_analysis(results: List[RunResult]) -> str:
    """Analyze temperature sweep results."""
    # Filter temperature sweep experiments
    temp_results = [r for r in results if r.experiment_type == "temperature_sweep"]

    # Group by model
    by_model = defaultdict(list)
    for r in temp_results:
        key = f"{r.model_size}_{r.notation}"
        by_model[key].append(r)

    report = ["## Temperature Analysis\n"]
    report.append("| Model | Temp | Valid% | Unique% | Constraint% |")
    report.append("|-------|------|--------|---------|-------------|")

    for model_key in sorted(by_model.keys()):
        runs = sorted(by_model[model_key], key=lambda x: x.temperature)
        for r in runs:
            report.append(
                f"| {model_key} | {r.temperature:.1f} | "
                f"{r.valid_rate*100:.1f}% | {r.diversity_rate*100:.1f}% | "
                f"{r.constraint_rate*100:.1f}% |"
            )

    # Summary insights
    report.append("\n### Key Insights\n")

    # Calculate averages by temperature
    temp_avgs = defaultdict(lambda: {"valid": [], "diverse": [], "constraint": []})
    for r in temp_results:
        temp_avgs[r.temperature]["valid"].append(r.valid_rate)
        temp_avgs[r.temperature]["diverse"].append(r.diversity_rate)
        temp_avgs[r.temperature]["constraint"].append(r.constraint_rate)

    report.append("**Average metrics by temperature:**\n")
    for temp in sorted(temp_avgs.keys()):
        avg_valid = sum(temp_avgs[temp]["valid"]) / len(temp_avgs[temp]["valid"])
        avg_diverse = sum(temp_avgs[temp]["diverse"]) / len(temp_avgs[temp]["diverse"])
        avg_constraint = sum(temp_avgs[temp]["constraint"]) / len(temp_avgs[temp]["constraint"])
        report.append(f"- **T={temp:.1f}**: Valid={avg_valid*100:.1f}%, Diverse={avg_diverse*100:.1f}%, Constraint={avg_constraint*100:.1f}%")

    return "\n".join(report)


def generate_model_comparison(results: List[RunResult]) -> str:
    """Compare models across configurations."""
    report = ["## Model Comparison\n"]

    # Group by model
    by_model = defaultdict(list)
    for r in results:
        key = f"{r.model_size}_{r.notation}"
        by_model[key].append(r)

    report.append("| Model | Runs | Avg Valid% | Avg Unique% | Avg Constraint% |")
    report.append("|-------|------|------------|-------------|-----------------|")

    for model_key in sorted(by_model.keys()):
        runs = by_model[model_key]
        avg_valid = sum(r.valid_rate for r in runs) / len(runs)
        avg_diverse = sum(r.diversity_rate for r in runs) / len(runs)
        avg_constraint = sum(r.constraint_rate for r in runs) / len(runs)

        report.append(
            f"| {model_key} | {len(runs)} | "
            f"{avg_valid*100:.1f}% | {avg_diverse*100:.1f}% | "
            f"{avg_constraint*100:.1f}% |"
        )

    # Compare infix vs prefix
    report.append("\n### Infix vs Prefix\n")

    infix_results = [r for r in results if r.notation == "infix"]
    prefix_results = [r for r in results if r.notation == "prefix"]

    if infix_results and prefix_results:
        infix_valid = sum(r.valid_rate for r in infix_results) / len(infix_results)
        prefix_valid = sum(r.valid_rate for r in prefix_results) / len(prefix_results)
        infix_diverse = sum(r.diversity_rate for r in infix_results) / len(infix_results)
        prefix_diverse = sum(r.diversity_rate for r in prefix_results) / len(prefix_results)

        report.append(f"- **Infix**: Valid={infix_valid*100:.1f}%, Diverse={infix_diverse*100:.1f}%")
        report.append(f"- **Prefix**: Valid={prefix_valid*100:.1f}%, Diverse={prefix_diverse*100:.1f}%")

        diff = infix_valid - prefix_valid
        winner = "Infix" if diff > 0 else "Prefix"
        report.append(f"\n**{winner}** notation has higher valid rate by {abs(diff)*100:.1f}pp")

    # Compare by model size
    report.append("\n### Model Size Effect\n")

    by_size = defaultdict(list)
    for r in results:
        by_size[r.model_size].append(r)

    for size in ["base", "medium", "large"]:
        if size in by_size:
            avg_valid = sum(r.valid_rate for r in by_size[size]) / len(by_size[size])
            avg_diverse = sum(r.diversity_rate for r in by_size[size]) / len(by_size[size])
            report.append(f"- **{size.capitalize()}**: Valid={avg_valid*100:.1f}%, Diverse={avg_diverse*100:.1f}%")

    return "\n".join(report)


def generate_prompt_analysis(results: List[RunResult]) -> str:
    """Analyze prompt variation results."""
    prompt_results = [r for r in results if r.experiment_type == "prompt_variation"]

    if not prompt_results:
        return "## Prompt Variation Analysis\n\nNo prompt variation experiments found.\n"

    report = ["## Prompt Variation Analysis\n"]

    # Group by prompt complexity
    by_vars = defaultdict(list)
    for r in prompt_results:
        vars_key = ",".join(sorted(r.vars))
        by_vars[vars_key].append(r)

    report.append("| Variables | Runs | Avg Valid% | Avg Unique% |")
    report.append("|-----------|------|------------|-------------|")

    for vars_key in sorted(by_vars.keys(), key=lambda x: len(x)):
        runs = by_vars[vars_key]
        avg_valid = sum(r.valid_rate for r in runs) / len(runs)
        avg_diverse = sum(r.diversity_rate for r in runs) / len(runs)
        report.append(f"| {vars_key} | {len(runs)} | {avg_valid*100:.1f}% | {avg_diverse*100:.1f}% |")

    return "\n".join(report)


def generate_sampling_analysis(results: List[RunResult]) -> str:
    """Analyze sampling method variations."""
    sampling_results = [r for r in results if r.experiment_type == "sampling_method"]

    if not sampling_results:
        return "## Sampling Method Analysis\n\nNo sampling experiments found.\n"

    report = ["## Sampling Method Analysis\n"]

    # Group by sampling config
    by_sampling = defaultdict(list)
    for r in sampling_results:
        key = f"top_p={r.top_p}, top_k={r.top_k}"
        by_sampling[key].append(r)

    report.append("| Sampling | Runs | Avg Valid% | Avg Unique% |")
    report.append("|----------|------|------------|-------------|")

    for sampling_key in sorted(by_sampling.keys()):
        runs = by_sampling[sampling_key]
        avg_valid = sum(r.valid_rate for r in runs) / len(runs)
        avg_diverse = sum(r.diversity_rate for r in runs) / len(runs)
        report.append(f"| {sampling_key} | {len(runs)} | {avg_valid*100:.1f}% | {avg_diverse*100:.1f}% |")

    return "\n".join(report)


def generate_full_report(results: List[RunResult]) -> str:
    """Generate comprehensive analysis report."""
    report = [
        "# Comprehensive Model Evaluation Report\n",
        f"**Total experiments analyzed:** {len(results)}\n",
        f"**Models evaluated:** {len(set(r.model for r in results))}\n",
        "",
    ]

    # Add each analysis section
    report.append(generate_model_comparison(results))
    report.append("")
    report.append(generate_temperature_analysis(results))
    report.append("")
    report.append(generate_prompt_analysis(results))
    report.append("")
    report.append(generate_sampling_analysis(results))

    # Overall conclusions
    report.append("\n## Conclusions\n")

    # Best model
    by_model = defaultdict(list)
    for r in results:
        key = f"{r.model_size}_{r.notation}"
        by_model[key].append(r.valid_rate)

    best_model = max(by_model.items(), key=lambda x: sum(x[1])/len(x[1]))
    report.append(f"1. **Best overall model:** {best_model[0]} (avg valid rate: {sum(best_model[1])/len(best_model[1])*100:.1f}%)")

    # Best temperature
    temp_avgs = defaultdict(list)
    for r in results:
        temp_avgs[r.temperature].append(r.valid_rate)

    # Balance between valid and diverse
    temp_scores = {}
    for temp, valids in temp_avgs.items():
        avg_valid = sum(valids) / len(valids)
        diverse_results = [r for r in results if r.temperature == temp]
        avg_diverse = sum(r.diversity_rate for r in diverse_results) / len(diverse_results)
        # Combined score (weight valid more)
        temp_scores[temp] = avg_valid * 0.6 + avg_diverse * 0.4

    best_temp = max(temp_scores.items(), key=lambda x: x[1])
    report.append(f"2. **Recommended temperature:** {best_temp[0]:.1f} (balanced valid/diverse)")

    # Notation recommendation
    infix_avg = sum(r.valid_rate for r in results if r.notation == "infix") / max(1, len([r for r in results if r.notation == "infix"]))
    prefix_avg = sum(r.valid_rate for r in results if r.notation == "prefix") / max(1, len([r for r in results if r.notation == "prefix"]))

    better_notation = "infix" if infix_avg > prefix_avg else "prefix"
    report.append(f"3. **Recommended notation:** {better_notation} ({max(infix_avg, prefix_avg)*100:.1f}% valid rate)")

    return "\n".join(report)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze comprehensive evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/comprehensive",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="COMPREHENSIVE_EVALUATION_REPORT.md",
        help="Output report file",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} experiments")

    print("Generating report...")
    report = generate_full_report(results)

    output_file = Path(args.output)
    output_file.write_text(report)
    print(f"Report saved to: {output_file}")

    # Print summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)

    by_model = defaultdict(list)
    for r in results:
        key = f"{r.model_size}_{r.notation}"
        by_model[key].append(r)

    for model_key in sorted(by_model.keys()):
        runs = by_model[model_key]
        avg_valid = sum(r.valid_rate for r in runs) / len(runs)
        avg_diverse = sum(r.diversity_rate for r in runs) / len(runs)
        print(f"{model_key:20} | Valid: {avg_valid*100:5.1f}% | Diverse: {avg_diverse*100:5.1f}%")


if __name__ == "__main__":
    main()
