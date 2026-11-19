"""
CLI commands for model evaluation.

Available commands:
- quality: Evaluate expression generation quality (valid rate, diversity)
- benchmark: Evaluate on symbolic regression benchmarks (R² scores)
- compare: Compare multiple evaluation runs
- report: Generate reports from evaluation results
"""

from .quality import execute_quality, add_quality_arguments
from .benchmark import execute_benchmark, add_benchmark_arguments, list_benchmarks
from .compare import execute_compare, add_compare_arguments, list_available_runs
from .report import execute_report, add_report_arguments
from .upload import execute_upload, execute_download, add_upload_arguments, add_download_arguments

__all__ = [
    "execute_quality",
    "add_quality_arguments",
    "execute_benchmark",
    "add_benchmark_arguments",
    "list_benchmarks",
    "execute_compare",
    "add_compare_arguments",
    "list_available_runs",
    "execute_report",
    "add_report_arguments",
    "execute_upload",
    "execute_download",
    "add_upload_arguments",
    "add_download_arguments",
]
