"""
Report generation command.

Generates detailed reports from evaluation runs:
- Markdown format for documentation
- HTML format for web viewing
- JSON format for programmatic access
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add 3_evaluation directory to path for imports
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from core.storage import ResultStorage

logger = logging.getLogger(__name__)


def generate_markdown_report(run_data: dict, samples: List[dict] = None) -> str:
    """Generate a detailed markdown report for a single run."""
    run_id = run_data["run_id"]
    config = run_data.get("config", {})
    metrics = run_data.get("metrics", {})

    lines = [
        f"# Evaluation Report: {run_id}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Configuration",
        "",
    ]

    # Model info
    if config:
        model_config = config.get("model", {})
        lines.append(f"**Model:** `{model_config.get('path', 'unknown')}`")
        lines.append("")

        # Generation settings
        gen_config = config.get("generation", {})
        lines.extend([
            "### Generation Parameters",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Temperature | {gen_config.get('temperature', 'N/A')} |",
            f"| Top-p | {gen_config.get('top_p', 'N/A')} |",
            f"| Top-k | {gen_config.get('top_k', 'N/A')} |",
            f"| Max tokens | {gen_config.get('max_new_tokens', 'N/A')} |",
            "",
        ])

        # Prompt settings
        prompt_config = config.get("prompt", {})
        if prompt_config:
            lines.extend([
                "### Prompt Configuration",
                "",
                f"- **Format:** {prompt_config.get('format', 'N/A')}",
                f"- **Variables:** {prompt_config.get('vars', [])}",
                f"- **Operators:** {prompt_config.get('ops', [])}",
                f"- **Constant:** {prompt_config.get('cons', 'N/A')}",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## Results Summary",
        "",
    ])

    # Main metrics table
    if metrics:
        lines.extend([
            "### Core Metrics",
            "",
            "| Metric | Value | Count |",
            "|--------|-------|-------|",
            f"| Valid Rate | {metrics.get('valid_rate', 0):.1%} | {metrics.get('valid_count', 0)} / {metrics.get('total_samples', 0)} |",
            f"| Parseable Rate | {metrics.get('parseable_rate', 0):.1%} | {metrics.get('parseable_count', 0)} / {metrics.get('total_samples', 0)} |",
            f"| Diversity Rate | {metrics.get('diversity_rate', 0):.1%} | {metrics.get('unique_count', 0)} unique |",
            f"| Constraint Adherence | {metrics.get('constraint_adherence_rate', 0):.1%} | {metrics.get('constraint_valid_count', 0)} / {metrics.get('total_samples', 0)} |",
            "",
        ])

        # Complexity statistics
        lines.extend([
            "### Complexity Statistics",
            "",
            f"- **Average complexity:** {metrics.get('avg_complexity', 0):.2f}",
            f"- **Min complexity:** {metrics.get('min_complexity', 0)}",
            f"- **Max complexity:** {metrics.get('max_complexity', 0)}",
            "",
            f"- **Average length:** {metrics.get('avg_length', 0):.1f} characters",
            f"- **Length range:** {metrics.get('min_length', 0)} - {metrics.get('max_length', 0)}",
            "",
        ])

        # Variable usage
        var_usage = metrics.get("variable_usage", {})
        if var_usage:
            lines.extend([
                "### Variable Usage",
                "",
                "| Variable | Count |",
                "|----------|-------|",
            ])
            for var, count in sorted(var_usage.items()):
                lines.append(f"| {var} | {count} |")
            lines.append("")

        # Operator usage
        op_usage = metrics.get("operator_usage", {})
        if op_usage:
            lines.extend([
                "### Operator Usage",
                "",
                "| Operator | Count |",
                "|----------|-------|",
            ])
            for op, count in sorted(op_usage.items(), key=lambda x: -x[1]):
                lines.append(f"| {op} | {count} |")
            lines.append("")

        # Error analysis
        error_types = metrics.get("error_types", {})
        if error_types:
            lines.extend([
                "### Error Analysis",
                "",
                "| Error Type | Count |",
                "|------------|-------|",
            ])
            for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                lines.append(f"| {error_type} | {count} |")
            lines.append("")

    # Sample expressions
    if samples:
        lines.extend([
            "---",
            "",
            "## Sample Expressions",
            "",
        ])

        # Valid samples
        valid_samples = [s for s in samples if s.get("valid")][:10]
        if valid_samples:
            lines.extend([
                "### Valid Expressions (sample)",
                "",
                "```",
            ])
            for s in valid_samples:
                lines.append(s.get("expression", ""))
            lines.extend([
                "```",
                "",
            ])

        # Invalid samples
        invalid_samples = [s for s in samples if not s.get("valid")][:5]
        if invalid_samples:
            lines.extend([
                "### Invalid Expressions (sample)",
                "",
            ])
            for s in invalid_samples:
                error = s.get("error", "unknown error")[:50]
                lines.append(f"- Error: {error}")
            lines.append("")

    lines.extend([
        "---",
        "",
        f"*Report generated by Seriguela Evaluation CLI*",
    ])

    return "\n".join(lines)


def generate_html_report(run_data: dict, samples: List[dict] = None) -> str:
    """Generate an HTML report for a single run."""
    run_id = run_data["run_id"]
    metrics = run_data.get("metrics", {})
    config = run_data.get("config", {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Evaluation Report: {run_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #4a90d9; margin-top: 30px; }}
        .metric-card {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4a90d9;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; }}
        .expression {{
            font-family: monospace;
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Report</h1>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('valid_rate', 0):.1%}</div>
                <div class="metric-label">Valid Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('diversity_rate', 0):.1%}</div>
                <div class="metric-label">Diversity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('unique_count', 0)}</div>
                <div class="metric-label">Unique</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('avg_complexity', 0):.1f}</div>
                <div class="metric-label">Avg Complexity</div>
            </div>
        </div>

        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Model</td><td>{config.get('model', {}).get('path', 'N/A')}</td></tr>
            <tr><td>Temperature</td><td>{config.get('generation', {}).get('temperature', 'N/A')}</td></tr>
            <tr><td>Top-p</td><td>{config.get('generation', {}).get('top_p', 'N/A')}</td></tr>
            <tr><td>Samples</td><td>{metrics.get('total_samples', 'N/A')}</td></tr>
        </table>

        <h2>Detailed Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Rate</th><th>Count</th></tr>
            <tr>
                <td>Valid expressions</td>
                <td>{metrics.get('valid_rate', 0):.1%}</td>
                <td>{metrics.get('valid_count', 0)} / {metrics.get('total_samples', 0)}</td>
            </tr>
            <tr>
                <td>Parseable expressions</td>
                <td>{metrics.get('parseable_rate', 0):.1%}</td>
                <td>{metrics.get('parseable_count', 0)} / {metrics.get('total_samples', 0)}</td>
            </tr>
            <tr>
                <td>Constraint adherence</td>
                <td>{metrics.get('constraint_adherence_rate', 0):.1%}</td>
                <td>{metrics.get('constraint_valid_count', 0)} / {metrics.get('total_samples', 0)}</td>
            </tr>
        </table>
"""

    # Add sample expressions if available
    if samples:
        valid_samples = [s for s in samples if s.get("valid")][:5]
        if valid_samples:
            html += """
        <h2>Sample Expressions</h2>
        <ul>
"""
            for s in valid_samples:
                expr = s.get("expression", "")
                html += f'            <li><span class="expression">{expr}</span></li>\n'
            html += "        </ul>\n"

    html += f"""
        <div class="footer">
            Report generated by Seriguela Evaluation CLI
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_json_report(run_data: dict, samples: List[dict] = None) -> str:
    """Generate a JSON report for a single run."""
    report = {
        "run_id": run_data["run_id"],
        "generated_at": datetime.now().isoformat(),
        "config": run_data.get("config"),
        "metrics": run_data.get("metrics"),
        "sample_count": run_data.get("sample_count", 0),
    }

    if samples:
        report["sample_expressions"] = [s.get("expression") for s in samples if s.get("valid")][:20]

    return json.dumps(report, indent=2)


def execute_report(args: argparse.Namespace):
    """
    Execute report generation.

    Args:
        args: Parsed command line arguments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    base_dir = getattr(args, "results_dir", "results")
    storage = ResultStorage(base_dir=base_dir)

    # Get run ID(s)
    run_ids = []
    if hasattr(args, "run") and args.run:
        run_ids = [args.run]
    elif hasattr(args, "runs") and args.runs:
        run_ids = args.runs

    if not run_ids:
        # List available runs
        print("\nAvailable runs:")
        for run_info in storage.list_runs_with_info():
            print(f"  - {run_info['run_id']}")
        print("\nUse --run <run_id> to generate a report.")
        return

    # Get format
    output_format = getattr(args, "format", "markdown")

    for run_id in run_ids:
        if not storage.run_exists(run_id):
            print(f"Run not found: {run_id}")
            continue

        print(f"\nGenerating {output_format} report for: {run_id}")

        # Load run data
        run_data = storage.load_run(run_id)

        # Load samples if needed
        samples = None
        if output_format in ["markdown", "html"]:
            try:
                samples = storage.load_samples(run_id, limit=50)
            except Exception:
                pass

        # Generate report
        if output_format == "markdown":
            report = generate_markdown_report(run_data, samples)
            ext = "md"
        elif output_format == "html":
            report = generate_html_report(run_data, samples)
            ext = "html"
        else:  # json
            report = generate_json_report(run_data, samples)
            ext = "json"

        # Determine output path
        if hasattr(args, "output") and args.output:
            output_path = args.output
        else:
            output_path = storage.get_run_dir(run_id) / f"report.{ext}"

        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Report saved to: {output_path}")

        # Also print markdown to console if short
        if output_format == "markdown" and len(report) < 3000:
            print("\n" + "=" * 60)
            print(report)


def add_report_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the report command."""
    parser.add_argument(
        "--run",
        type=str,
        help="Run ID to generate report for",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Multiple run IDs to generate reports for",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: results/<run_id>/report.<ext>)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results (default: results)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    add_report_arguments(parser)
    args = parser.parse_args()
    execute_report(args)
