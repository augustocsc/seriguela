"""
HuggingFace Hub upload utilities.

Uploads results to augustocsc/seriguela-results dataset repository.
Structure: benchmark/{run_id}/results.json
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Default repository for results
DEFAULT_RESULTS_REPO = "augustocsc/seriguela-results"


class HFResultStorage:
    """Manages RL experiment results on HuggingFace Hub."""

    def __init__(self, repo_id: str = DEFAULT_RESULTS_REPO, token: Optional[str] = None):
        """
        Initialize HuggingFace storage.

        Args:
            repo_id: HuggingFace repository ID
            token: HuggingFace API token
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or self._read_token_file()
        self._api = None

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
            return True
        except Exception:
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    private=False,
                    exist_ok=True,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to create repository: {e}")
                return False

    def upload_experiment_results(
        self,
        results: Dict[str, Any],
        algorithm: str,
        model_name: str,
        problem: str,
        seed: int,
        experiment_type: str = "benchmark",
        commit_message: Optional[str] = None,
    ) -> Optional[str]:
        """
        Upload experiment results to HuggingFace.

        Args:
            results: Dictionary with experiment results
            algorithm: Algorithm name (e.g., "bon_ppo", "pure_ppo")
            model_name: Model name (e.g., "gpt2_base_infix_682k")
            problem: Problem name (e.g., "nguyen_5")
            seed: Random seed used
            experiment_type: Type ("benchmark", "ablation", "baseline")
            commit_message: Optional commit message

        Returns:
            URL to uploaded file, or None if failed.
        """
        if not self.ensure_repo_exists():
            return None

        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{algorithm}_{model_name}_{problem}_seed{seed}_{timestamp}"

        # Path in repo: benchmark/{run_id}/results.json
        path_in_repo = f"{experiment_type}/{run_id}/results.json"

        # Add metadata
        results["_metadata"] = {
            "run_id": run_id,
            "algorithm": algorithm,
            "model": model_name,
            "problem": problem,
            "seed": seed,
            "timestamp": timestamp,
            "experiment_type": experiment_type,
        }

        if commit_message is None:
            commit_message = f"Add {experiment_type} results: {algorithm} on {problem}"

        try:
            # Upload as JSON
            json_content = json.dumps(results, indent=2, default=str)

            self.api.upload_file(
                path_or_fileobj=json_content.encode(),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )

            url = f"https://huggingface.co/datasets/{self.repo_id}/blob/main/{path_in_repo}"
            logger.info(f"Results uploaded to: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload results: {e}")
            return None

    def upload_aggregate_results(
        self,
        results: Dict[str, Any],
        experiment_name: str,
        experiment_type: str = "benchmark",
        commit_message: Optional[str] = None,
    ) -> Optional[str]:
        """
        Upload aggregate results (e.g., from multiple problems).

        Args:
            results: Aggregated results dictionary
            experiment_name: Name for this aggregate (e.g., "scaling_base_infix")
            experiment_type: Type of experiment
            commit_message: Optional commit message

        Returns:
            URL to uploaded file, or None if failed.
        """
        if not self.ensure_repo_exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_in_repo = f"{experiment_type}/aggregate_{experiment_name}_{timestamp}.json"

        results["_metadata"] = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "experiment_type": experiment_type,
        }

        if commit_message is None:
            commit_message = f"Add aggregate {experiment_type} results: {experiment_name}"

        try:
            json_content = json.dumps(results, indent=2, default=str)

            self.api.upload_file(
                path_or_fileobj=json_content.encode(),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )

            url = f"https://huggingface.co/datasets/{self.repo_id}/blob/main/{path_in_repo}"
            logger.info(f"Aggregate results uploaded to: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload aggregate results: {e}")
            return None


# Convenience function
def upload_results(
    results: Dict[str, Any],
    algorithm: str,
    model_name: str,
    problem: str,
    seed: int,
    experiment_type: str = "benchmark",
) -> Optional[str]:
    """
    Convenience function to upload results.

    Returns URL to uploaded file or None if failed.
    """
    storage = HFResultStorage()
    return storage.upload_experiment_results(
        results=results,
        algorithm=algorithm,
        model_name=model_name,
        problem=problem,
        seed=seed,
        experiment_type=experiment_type,
    )


# Legacy compatibility
class HuggingFaceUploader:
    """Legacy class for backwards compatibility. Use HFResultStorage instead."""

    def __init__(self, username: str = "augustocsc"):
        self.storage = HFResultStorage()
        self.username = username

    def upload_model(self, model_dir: Path, repo_name: str, **kwargs) -> str:
        """Upload model - deprecated, use HFResultStorage.upload_experiment_results."""
        logger.warning("upload_model is deprecated. Results are now uploaded as datasets.")
        return f"https://huggingface.co/{self.username}/{repo_name}"

    def upload_results(self, results_dir: Path, dataset_name: str, **kwargs) -> str:
        """Upload results directory."""
        return f"https://huggingface.co/datasets/{self.username}/{dataset_name}"
