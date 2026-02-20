"""
Compare runs command.

Compares multiple evaluation runs:
- Side-by-side metrics comparison
- Statistical significance tests (optional)
- Output as table or markdown
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional

# Add 3_evaluation directory to path for imports
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from core.storage import ResultStorage
from core.metrics import QualityMetrics

logger = logging.getLogger(__name__)


def format_percentage(value: float) -> str:
    """Format a float as percentage string."""
    return f"{value:.1%}"


def format_float(value: float, decimals: int = 2) -> str:
    """Format a float with specified decimals."""
    return f"{value:.{decimals}f}"


def create_comparison_table(runs_data: List[dict]) -> str:
    """Create a comparison table in text format."""
    if not runs_data:
        return "No data to compare."

    # Define columns
    columns = [
        ("Run ID", lambda d: d["run_id"][:20]),
        ("Model", lambda d: d.get("model", "unknown")[:25]),
        ("Samples", lambda d: str(d["metrics"].get("total_samples", 0))),
        ("Valid", lambda d: format_percentage(d["metrics"].get("valid_rate", 0))),
        ("Diverse", lambda d: format_percentage(d["metrics"].get("diversity_rate", 0))),
        ("Constraint", lambda d: format_percentage(d["metrics"].get("constraint_adherence_rate", 0))),
        ("Complexity", lambda d: format_float(d["metrics"].get("avg_complexity", 0), 1)),
    ]

    # Calculate column widths
    widths = []
    for name, _ in columns:
        max_width = len(name)
        for data in runs_data:
            try:
                value = columns[columns.index((name, _))][1](data)
                max_width = max(max_width, len(str(value)))
            except Exception:
                pass
        widths.append(max_width + 2)

    # Build table
    lines = []

    # Header
    header = "|".join(name.center(widths[i]) for i, (name, _) in enumerate(columns))
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for data in runs_data:
        row_values = []
        for i, (name, getter) in enumerate(columns):
            try:
                value = getter(data)
            except Exception:
                value = "N/A"
            row_values.append(str(value).center(widths[i]))
        lines.append("|".join(row_values))

    return "\n".join(lines)


def create_comparison_markdown(runs_data: List[dict], output_path: Optional[str] = None) -> str:
    """Create a comparison table in markdown format."""
    if not runs_data:
        return "No data to compare."

    lines = [
        "# Evaluation Run Comparison",
        "",
        f"Comparing {len(runs_data)} runs.",
        "",
        "## Summary Table",
        "",
        "| Run ID | Model | Samples | Valid Rate | Diversity | Constraint | Avg Complexity |",
        "|--------|-------|---------|------------|-----------|------------|----------------|",
    ]

    for data in runs_data:
        run_id = data["run_id"][:15]
        model = data.get("model", "unknown")[:20]
        metrics = data["metrics"]

        row = (
            f"| {run_id} "
            f"| {model} "
            f"| {metrics.get('total_samples', 0)} "
            f"| {format_percentage(metrics.get('valid_rate', 0))} "
            f"| {format_percentage(metrics.get('diversity_rate', 0))} "
            f"| {format_percentage(metrics.get('constraint_adherence_rate', 0))} "
            f"| {format_float(metrics.get('avg_complexity', 0), 1)} |"
        )
        lines.append(row)

    lines.extend([
        "",
        "## Detailed Comparison",
        "",
    ])

    # Add detailed info for each run
    for data in runs_data:
        lines.append(f"### {data['run_id']}")
        lines.append("")

        if data.get("config"):
            config = data["config"]
            lines.append("**Configuration:**")
            lines.append(f"- Model: `{config.get('model', {}).get('path', 'unknown')}`")
            gen = config.get("generation", {})
            lines.append(f"- Temperature: {gen.get('temperature', 'N/A')}")
            lines.append(f"- Top-p: {gen.get('top_p', 'N/A')}")
            lines.append("")

        metrics = data["metrics"]
        lines.append("**Metrics:**")
        lines.append(f"- Valid expressions: {metrics.get('valid_count', 0)} / {metrics.get('total_samples', 0)}")
        lines.append(f"- Unique expressions: {metrics.get('unique_count', 0)}")
        lines.append(f"- Complexity range: {metrics.get('min_complexity', 0)} - {metrics.get('max_complexity', 0)}")

        if metrics.get("error_types"):
            lines.append("")
            lines.append("**Error distribution:**")
            for error_type, count in sorted(metrics["error_types"].items(), key=lambda x: -x[1])[:5]:
                lines.append(f"- {error_type}: {count}")

        lines.append("")

    # Find best performers
    if len(runs_data) > 1:
        lines.append("## Best Performers")
        lines.append("")

        valid_rates = [(d["run_id"], d["metrics"].get("valid_rate", 0)) for d in runs_data]
        diversity_rates = [(d["run_id"], d["metrics"].get("diversity_rate", 0)) for d in runs_data]

        best_valid = max(valid_rates, key=lambda x: x[1])
        best_diverse = max(diversity_rates, key=lambda x: x[1])

        lines.append(f"- **Highest valid rate:** {best_valid[0]} ({format_percentage(best_valid[1])})")
        lines.append(f"- **Highest diversity:** {best_diverse[0]} ({format_percentage(best_diverse[1])})")

    markdown = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(markdown)
        print(f"Comparison saved to: {output_path}")

    return markdown


def execute_compare(args: argparse.Namespace):
    """
    Execute comparison of runs.

    Args:
        args: Parsed command line arguments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Get runs to compare
    run_ids = args.runs
    base_dir = getattr(args, "results_dir", "results")

    storage = ResultStorage(base_dir=base_dir)

    print(f"\n{'='*60}")
    print("Comparing Evaluation Runs")
    print(f"{'='*60}\n")

    # Load data for each run
    runs_data = []
    for run_id in run_ids:
        if not storage.run_exists(run_id):
            print(f"Warning: Run '{run_id}' not found, skipping.")
            continue

        try:
            data = storage.load_run(run_id)

            # Extract model name from config
            if data["config"]:
                data["model"] = data["config"].get("model", {}).get("path", "unknown")
            else:
                data["model"] = "unknown"

            runs_data.append(data)
            print(f"Loaded: {run_id}")
        except Exception as e:
            print(f"Error loading {run_id}: {e}")

    if not runs_data:
        print("\nNo valid runs to compare.")
        return

    print(f"\nComparing {len(runs_data)} runs:\n")

    # Create and display comparison table
    table = create_comparison_table(runs_data)
    print(table)

    # Generate markdown if output specified
    output_path = getattr(args, "output", None)
    if output_path:
        create_comparison_markdown(runs_data, output_path)
    else:
        print("\n(Use --output to save comparison as markdown)")

    # Summary
    print(f"\n{'='*60}")
    if len(runs_data) > 1:
        valid_rates = [(d["run_id"], d["metrics"].get("valid_rate", 0)) for d in runs_data]
        best_valid = max(valid_rates, key=lambda x: x[1])
        print(f"Best valid rate: {best_valid[0]} ({format_percentage(best_valid[1])})")


def add_compare_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the compare command."""
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run IDs to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for comparison (markdown format)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results (default: results)",
    )


def list_available_runs(args: argparse.Namespace):
    """List all available runs."""
    base_dir = getattr(args, "results_dir", "results")
    storage = ResultStorage(base_dir=base_dir)

    runs = storage.list_runs_with_info()

    if not runs:
        print("No runs found.")
        return

    print(f"\nAvailable runs ({len(runs)} total):\n")
    print(f"{'Run ID':<35} {'Model':<30} {'Valid Rate':<12} {'Samples'}")
    print("-" * 90)

    for run in runs:
        run_id = run.get("run_id", "unknown")
        model = run.get("model", "unknown")[:28]
        valid_rate = run.get("valid_rate", 0)
        samples = run.get("total_samples", 0)

        print(f"{run_id:<35} {model:<30} {valid_rate:>10.1%} {samples:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    add_compare_arguments(parser)
    args = parser.parse_args()
    execute_compare(args)
