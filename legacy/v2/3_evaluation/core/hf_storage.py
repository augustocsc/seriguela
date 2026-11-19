"""
HuggingFace storage for evaluation results.

Uploads and retrieves evaluation results from HuggingFace Hub.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default repository for results
DEFAULT_RESULTS_REPO = "augustocsc/seriguela-results"


class HFResultStorage:
    """Manages evaluation results on HuggingFace Hub."""

    def __init__(self, repo_id: str = DEFAULT_RESULTS_REPO, token: Optional[str] = None):
        """
        Initialize HuggingFace storage.

        Args:
            repo_id: HuggingFace repository ID (e.g., "username/repo-name")
            token: HuggingFace API token (reads from env if not provided)
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or self._read_token_file()

        # Lazy import to avoid issues if huggingface_hub not installed
        self._api = None
        self._fs = None

    def _read_token_file(self) -> Optional[str]:
        """Read token from ~/.tokens.txt file."""
        token_file = Path.home() / ".tokens.txt"
        if token_file.exists():
            try:
                with open(token_file, "r") as f:
                    for line in f:
                        if line.strip().startswith("huggingface"):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                return parts[1].strip()
            except Exception as e:
                logger.warning(f"Could not read token file: {e}")
        return None

    @property
    def api(self):
        """Lazy load HuggingFace API."""
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def ensure_repo_exists(self) -> bool:
        """Create the repository if it doesn't exist."""
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="dataset")
            logger.info(f"Repository {self.repo_id} exists")
            return True
        except Exception:
            logger.info(f"Creating repository {self.repo_id}")
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    private=False,
                    exist_ok=True,
                )
                # Create initial README
                readme_content = """---
license: mit
task_categories:
  - tabular-regression
tags:
  - symbolic-regression
  - evaluation-results
  - seriguela
---

# Seriguela Evaluation Results

This dataset contains evaluation results for symbolic regression models trained in the Seriguela project.

## Structure

- `quality/` - Generation quality evaluation results (valid rate, diversity, etc.)
- `benchmark/` - Benchmark evaluation results (R² scores on Nguyen benchmarks)

## Usage

```python
from datasets import load_dataset

# Load all results
ds = load_dataset("augustocsc/seriguela-results")

# Or load specific files
from huggingface_hub import hf_hub_download
metrics = hf_hub_download(
    repo_id="augustocsc/seriguela-results",
    filename="quality/run_xxx/metrics.json",
    repo_type="dataset"
)
```

## Related

- Models: [augustocsc/gpt2_base_infix_682k](https://huggingface.co/augustocsc/gpt2_base_infix_682k)
- Code: [github.com/augustocsc/seriguela](https://github.com/augustocsc/seriguela)
"""
                self.api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
                return True
            except Exception as e:
                logger.error(f"Failed to create repository: {e}")
                return False

    def upload_run(
        self,
        run_dir: Path,
        eval_type: str = "quality",
        commit_message: Optional[str] = None,
    ) -> bool:
        """
        Upload a single evaluation run to HuggingFace.

        Args:
            run_dir: Path to the run directory (e.g., results/quality/run_xxx)
            eval_type: Type of evaluation ("quality" or "benchmark")
            commit_message: Optional commit message

        Returns:
            True if upload successful, False otherwise.
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            logger.error(f"Run directory not found: {run_dir}")
            return False

        run_id = run_dir.name

        # Ensure repo exists
        if not self.ensure_repo_exists():
            return False

        # Files to upload
        files_to_upload = []
        for file in run_dir.iterdir():
            if file.is_file() and file.suffix in [".json", ".yaml", ".txt", ".jsonl"]:
                files_to_upload.append(file)

        if not files_to_upload:
            logger.warning(f"No files to upload in {run_dir}")
            return False

        # Upload files
        if commit_message is None:
            commit_message = f"Add {eval_type} results: {run_id}"

        try:
            for file in files_to_upload:
                path_in_repo = f"{eval_type}/{run_id}/{file.name}"
                logger.info(f"Uploading {file.name} to {path_in_repo}")

                self.api.upload_file(
                    path_or_fileobj=str(file),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=commit_message,
                )

            logger.info(f"Successfully uploaded run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload run: {e}")
            return False

    def upload_directory(
        self,
        local_dir: Path,
        eval_type: str = "quality",
        commit_message: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Upload all runs from a local directory.

        Args:
            local_dir: Path to results directory (e.g., results/quality)
            eval_type: Type of evaluation
            commit_message: Optional commit message

        Returns:
            Dictionary mapping run_id to upload success status.
        """
        local_dir = Path(local_dir)
        results = {}

        if not local_dir.exists():
            logger.error(f"Directory not found: {local_dir}")
            return results

        for run_dir in sorted(local_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                success = self.upload_run(run_dir, eval_type, commit_message)
                results[run_dir.name] = success

        return results

    def download_run(
        self,
        run_id: str,
        eval_type: str = "quality",
        local_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Download a single run from HuggingFace.

        Args:
            run_id: Run ID to download
            eval_type: Type of evaluation
            local_dir: Local directory to save to (default: results/{eval_type})

        Returns:
            Path to downloaded run directory, or None if failed.
        """
        from huggingface_hub import hf_hub_download, list_repo_files

        if local_dir is None:
            local_dir = Path(f"results/{eval_type}")
        local_dir = Path(local_dir)

        run_dir = local_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # List files in the run directory
            all_files = list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
            )

            prefix = f"{eval_type}/{run_id}/"
            run_files = [f for f in all_files if f.startswith(prefix)]

            if not run_files:
                logger.warning(f"No files found for run {run_id}")
                return None

            for remote_file in run_files:
                local_file = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_file,
                    repo_type="dataset",
                    token=self.token,
                    local_dir=str(local_dir.parent.parent),
                )
                logger.info(f"Downloaded {remote_file}")

            return run_dir

        except Exception as e:
            logger.error(f"Failed to download run {run_id}: {e}")
            return None

    def list_runs(self, eval_type: str = "quality") -> List[str]:
        """
        List all available runs on HuggingFace.

        Args:
            eval_type: Type of evaluation to list

        Returns:
            List of run IDs.
        """
        from huggingface_hub import list_repo_files

        try:
            all_files = list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
            )

            # Extract unique run IDs
            prefix = f"{eval_type}/"
            runs = set()
            for f in all_files:
                if f.startswith(prefix):
                    parts = f[len(prefix):].split("/")
                    if parts and parts[0].startswith("run_"):
                        runs.add(parts[0])

            return sorted(runs, reverse=True)

        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []

    def sync_all(
        self,
        local_base_dir: Path = Path("results"),
        upload: bool = True,
        download: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronize local results with HuggingFace.

        Args:
            local_base_dir: Base directory for results
            upload: Upload local runs not on HF
            download: Download HF runs not local

        Returns:
            Summary of sync operations.
        """
        summary = {"uploaded": [], "downloaded": [], "errors": []}

        for eval_type in ["quality", "benchmark"]:
            local_dir = local_base_dir / eval_type

            # Get local runs
            local_runs = set()
            if local_dir.exists():
                for d in local_dir.iterdir():
                    if d.is_dir() and d.name.startswith("run_"):
                        local_runs.add(d.name)

            # Get remote runs
            remote_runs = set(self.list_runs(eval_type))

            # Upload local-only runs
            if upload:
                to_upload = local_runs - remote_runs
                for run_id in to_upload:
                    success = self.upload_run(local_dir / run_id, eval_type)
                    if success:
                        summary["uploaded"].append(f"{eval_type}/{run_id}")
                    else:
                        summary["errors"].append(f"upload:{eval_type}/{run_id}")

            # Download remote-only runs
            if download:
                to_download = remote_runs - local_runs
                for run_id in to_download:
                    result = self.download_run(run_id, eval_type, local_dir)
                    if result:
                        summary["downloaded"].append(f"{eval_type}/{run_id}")
                    else:
                        summary["errors"].append(f"download:{eval_type}/{run_id}")

        return summary
