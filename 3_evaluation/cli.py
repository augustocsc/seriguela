#!/usr/bin/env python3
"""
Seriguela Model Evaluation CLI

Unified command-line interface for evaluating symbolic regression models.

Usage:
    # Quality evaluation (generation phase)
    python -m 3_evaluation.cli quality --model augustocsc/gpt2_large_infix_682k --num-samples 500

    # Benchmark evaluation (RL phase)
    python -m 3_evaluation.cli benchmark --model augustocsc/gpt2_large_infix_682k --benchmark nguyen_5

    # Compare runs
    python -m 3_evaluation.cli compare --runs run_001 run_002

    # Generate report
    python -m 3_evaluation.cli report --run run_001 --format markdown

    # List runs
    python -m 3_evaluation.cli list
"""

import sys
import argparse
from pathlib import Path

# Ensure proper imports when running as module
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from commands.quality import execute_quality, add_quality_arguments
from commands.benchmark import execute_benchmark, add_benchmark_arguments, list_benchmarks
from commands.compare import execute_compare, add_compare_arguments, list_available_runs
from commands.report import execute_report, add_report_arguments
from core.storage import ResultStorage


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Seriguela Model Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quality evaluation (generation phase)
  evaluate quality --model augustocsc/gpt2_large_infix_682k --num-samples 500

  # Benchmark evaluation (RL phase)
  evaluate benchmark --model augustocsc/gpt2_large_infix_682k --benchmark nguyen_5

  # Compare and report
  evaluate compare --runs run_001 run_002
  evaluate report --run run_001 --format markdown
  evaluate list

For more information, see: https://github.com/augustocsc/seriguela
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Seriguela Evaluate CLI v1.0.0",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        help="Use 'evaluate <command> --help' for more info",
    )

    # Subcommand: quality
    quality_parser = subparsers.add_parser(
        "quality",
        help="Evaluate expression generation quality",
        description="Generate expressions and evaluate quality metrics (valid rate, diversity, etc.)",
    )
    add_quality_arguments(quality_parser)

    # Subcommand: benchmark
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate on symbolic regression benchmarks",
        description="Generate candidate expressions and calculate R² scores on benchmarks (Nguyen, etc.)",
    )
    add_benchmark_arguments(benchmark_parser)

    # Subcommand: compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare evaluation runs",
        description="Compare metrics from multiple evaluation runs.",
    )
    add_compare_arguments(compare_parser)

    # Subcommand: report
    report_parser = subparsers.add_parser(
        "report",
        help="Generate reports",
        description="Generate detailed reports from evaluation results.",
    )
    add_report_arguments(report_parser)

    # Subcommand: list
    list_parser = subparsers.add_parser(
        "list",
        help="List available runs",
        description="Show all available evaluation runs.",
    )
    list_parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing results (default: results/quality and results/benchmark)",
    )
    list_parser.add_argument(
        "--type",
        type=str,
        choices=["quality", "benchmark", "all"],
        default="all",
        help="Type of runs to list (default: all)",
    )

    # Subcommand: benchmarks (list available benchmarks)
    benchmarks_parser = subparsers.add_parser(
        "benchmarks",
        help="List available benchmarks",
        description="Show all available benchmark problems.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle no command
    if args.command is None:
        parser.print_help()
        print("\nUse 'evaluate <command> --help' for command-specific help.")
        return

    # Execute command
    if args.command == "quality":
        execute_quality(args)
    elif args.command == "benchmark":
        execute_benchmark(args)
    elif args.command == "compare":
        execute_compare(args)
    elif args.command == "report":
        execute_report(args)
    elif args.command == "list":
        _list_runs(args)
    elif args.command == "benchmarks":
        list_benchmarks()
    else:
        parser.print_help()


def _list_runs(args):
    """List runs from quality and/or benchmark directories."""
    run_type = getattr(args, "type", "all")
    results_dir = getattr(args, "results_dir", None)

    dirs_to_check = []
    if run_type in ["quality", "all"]:
        dirs_to_check.append(("quality", results_dir or "results/quality"))
    if run_type in ["benchmark", "all"]:
        dirs_to_check.append(("benchmark", results_dir or "results/benchmark"))

    total_runs = 0
    for eval_type, dir_path in dirs_to_check:
        storage = ResultStorage(base_dir=dir_path)
        runs = storage.list_runs_with_info()

        if runs:
            print(f"\n{eval_type.upper()} runs ({dir_path}):")
            print("-" * 90)

            if eval_type == "quality":
                print(f"{'Run ID':<35} {'Model':<25} {'Valid Rate':<12} {'Samples'}")
                print("-" * 90)
                for run in runs:
                    run_id = run.get("run_id", "unknown")
                    model = run.get("model", "unknown")[:23]
                    valid_rate = run.get("valid_rate", 0)
                    samples = run.get("total_samples", 0)
                    print(f"{run_id:<35} {model:<25} {valid_rate:>10.1%} {samples:>8}")
            else:  # benchmark
                print(f"{'Run ID':<35} {'Model':<20} {'Benchmark':<12} {'Best R²':<10} {'Samples'}")
                print("-" * 90)
                for run in runs:
                    run_id = run.get("run_id", "unknown")
                    model = run.get("model", "unknown")[:18]
                    benchmark = run.get("benchmark", "unknown")[:10]
                    best_r2 = run.get("best_r2")
                    r2_str = f"{best_r2:.4f}" if best_r2 is not None else "N/A"
                    samples = run.get("total_samples", 0)
                    print(f"{run_id:<35} {model:<20} {benchmark:<12} {r2_str:<10} {samples:>8}")

            total_runs += len(runs)

    if total_runs == 0:
        print("\nNo runs found.")
        print("Run 'evaluate quality' or 'evaluate benchmark' to create runs.")


if __name__ == "__main__":
    main()
