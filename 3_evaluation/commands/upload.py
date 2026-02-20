"""
Upload command for evaluation results.

Uploads evaluation results to HuggingFace Hub for persistent storage.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add paths for imports
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from core.hf_storage import HFResultStorage, DEFAULT_RESULTS_REPO

logger = logging.getLogger(__name__)


def execute_upload(args: argparse.Namespace):
    """
    Execute upload command.

    Args:
        args: Parsed command line arguments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print(f"\n{'='*60}")
    print("Seriguela Results Upload")
    print(f"{'='*60}")
    print(f"Repository: {args.repo}")
    print(f"{'='*60}\n")

    # Initialize storage
    storage = HFResultStorage(repo_id=args.repo)

    if args.run:
        # Upload specific run
        eval_type = args.type
        run_dir = Path(args.results_dir) / eval_type / args.run

        if not run_dir.exists():
            # Try to find the run
            for et in ["quality", "benchmark"]:
                test_dir = Path(args.results_dir) / et / args.run
                if test_dir.exists():
                    run_dir = test_dir
                    eval_type = et
                    break

        if not run_dir.exists():
            print(f"Error: Run not found: {args.run}")
            print(f"Searched in: {args.results_dir}/quality/ and {args.results_dir}/benchmark/")
            return

        print(f"Uploading run: {args.run}")
        print(f"Type: {eval_type}")
        print(f"Directory: {run_dir}")

        success = storage.upload_run(run_dir, eval_type)

        if success:
            print(f"\n✓ Successfully uploaded {args.run}")
            print(f"View at: https://huggingface.co/datasets/{args.repo}/tree/main/{eval_type}/{args.run}")
        else:
            print(f"\n✗ Failed to upload {args.run}")

    elif args.sync:
        # Sync all results
        print("Synchronizing results with HuggingFace...")
        print(f"Local directory: {args.results_dir}")

        summary = storage.sync_all(
            local_base_dir=Path(args.results_dir),
            upload=True,
            download=args.download,
        )

        print(f"\n{'='*60}")
        print("Sync Summary")
        print(f"{'='*60}")

        if summary["uploaded"]:
            print(f"\nUploaded ({len(summary['uploaded'])}):")
            for item in summary["uploaded"]:
                print(f"  ✓ {item}")

        if summary["downloaded"]:
            print(f"\nDownloaded ({len(summary['downloaded'])}):")
            for item in summary["downloaded"]:
                print(f"  ✓ {item}")

        if summary["errors"]:
            print(f"\nErrors ({len(summary['errors'])}):")
            for item in summary["errors"]:
                print(f"  ✗ {item}")

        if not summary["uploaded"] and not summary["downloaded"]:
            print("\nNo changes - local and remote are in sync.")

    else:
        # Upload all from specified type
        eval_type = args.type
        local_dir = Path(args.results_dir) / eval_type

        if not local_dir.exists():
            print(f"Error: Directory not found: {local_dir}")
            return

        print(f"Uploading all {eval_type} runs from: {local_dir}")

        results = storage.upload_directory(local_dir, eval_type)

        print(f"\n{'='*60}")
        print("Upload Summary")
        print(f"{'='*60}")

        success_count = sum(1 for v in results.values() if v)
        fail_count = sum(1 for v in results.values() if not v)

        for run_id, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {run_id}")

        print(f"\nTotal: {success_count} uploaded, {fail_count} failed")

        if success_count > 0:
            print(f"\nView at: https://huggingface.co/datasets/{args.repo}/tree/main/{eval_type}")


def execute_download(args: argparse.Namespace):
    """
    Execute download command.

    Args:
        args: Parsed command line arguments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print(f"\n{'='*60}")
    print("Seriguela Results Download")
    print(f"{'='*60}")
    print(f"Repository: {args.repo}")
    print(f"{'='*60}\n")

    storage = HFResultStorage(repo_id=args.repo)

    if args.run:
        # Download specific run
        print(f"Downloading run: {args.run}")

        result = storage.download_run(
            run_id=args.run,
            eval_type=args.type,
            local_dir=Path(args.results_dir) / args.type,
        )

        if result:
            print(f"\n✓ Downloaded to: {result}")
        else:
            print(f"\n✗ Failed to download {args.run}")

    elif args.list:
        # List available runs
        print("Available runs on HuggingFace:\n")

        for eval_type in ["quality", "benchmark"]:
            runs = storage.list_runs(eval_type)
            if runs:
                print(f"{eval_type.upper()} runs:")
                for run_id in runs:
                    print(f"  - {run_id}")
                print()

        if not storage.list_runs("quality") and not storage.list_runs("benchmark"):
            print("No runs found.")

    else:
        # Download all
        print("Downloading all results...")

        summary = storage.sync_all(
            local_base_dir=Path(args.results_dir),
            upload=False,
            download=True,
        )

        if summary["downloaded"]:
            print(f"\nDownloaded {len(summary['downloaded'])} runs:")
            for item in summary["downloaded"]:
                print(f"  ✓ {item}")
        else:
            print("\nNo new runs to download - local is up to date.")


def add_upload_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the upload command."""
    parser.add_argument(
        "--run",
        type=str,
        help="Specific run ID to upload (e.g., run_20260220_052016_f5dcb0)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["quality", "benchmark"],
        default="quality",
        help="Type of evaluation results (default: quality)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync all local results with HuggingFace",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Also download remote runs not present locally (use with --sync)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_RESULTS_REPO,
        help=f"HuggingFace repository (default: {DEFAULT_RESULTS_REPO})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Local results directory (default: results)",
    )


def add_download_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the download command."""
    parser.add_argument(
        "--run",
        type=str,
        help="Specific run ID to download",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["quality", "benchmark"],
        default="quality",
        help="Type of evaluation results (default: quality)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available runs on HuggingFace",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_RESULTS_REPO,
        help=f"HuggingFace repository (default: {DEFAULT_RESULTS_REPO})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Local results directory (default: results)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload results to HuggingFace")
    add_upload_arguments(parser)
    args = parser.parse_args()
    execute_upload(args)
