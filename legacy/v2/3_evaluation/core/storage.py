"""
Result storage utilities.

Manages persistence of evaluation results:
- Creates run directories with unique IDs
- Saves configuration, samples, and metrics
- Loads and lists previous runs
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib
import yaml

from .metrics import QualityMetrics, SampleResult

logger = logging.getLogger(__name__)


class ResultStorage:
    """Manages persistence of evaluation results."""

    def __init__(self, base_dir: str = "results"):
        """
        Initialize the storage.

        Args:
            base_dir: Base directory for storing results.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _generate_run_id(self, config: dict) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create short hash from config
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]

        return f"run_{timestamp}_{config_hash}"

    def create_run(self, config: dict) -> str:
        """
        Create a new run and save its configuration.

        Args:
            config: Run configuration dictionary.

        Returns:
            Run ID string.
        """
        run_id = self._generate_run_id(config)
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata to config
        full_config = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            **config,
        }

        # Save config as YAML
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created run: {run_id}")
        return run_id

    def get_run_dir(self, run_id: str) -> Path:
        """Get the directory for a run."""
        return self.base_dir / run_id

    def save_sample(self, run_id: str, sample: SampleResult):
        """
        Save a single sample result (append to JSONL file).

        Args:
            run_id: The run ID.
            sample: The sample result to save.
        """
        run_dir = self.base_dir / run_id
        samples_path = run_dir / "samples.jsonl"

        with open(samples_path, "a") as f:
            f.write(json.dumps(sample.to_dict()) + "\n")

    def save_samples_batch(self, run_id: str, samples: List[SampleResult]):
        """
        Save multiple samples at once.

        Args:
            run_id: The run ID.
            samples: List of sample results to save.
        """
        run_dir = self.base_dir / run_id
        samples_path = run_dir / "samples.jsonl"

        with open(samples_path, "a") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

    def save_metrics(self, run_id: str, metrics: QualityMetrics):
        """
        Save final metrics for a run.

        Args:
            run_id: The run ID.
            metrics: The metrics to save.
        """
        run_dir = self.base_dir / run_id
        metrics_path = run_dir / "metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Saved metrics for run: {run_id}")

    def save_summary(self, run_id: str, metrics: QualityMetrics):
        """
        Save a human-readable summary.

        Args:
            run_id: The run ID.
            metrics: The metrics to summarize.
        """
        run_dir = self.base_dir / run_id
        summary_path = run_dir / "summary.txt"

        with open(summary_path, "w") as f:
            f.write(metrics.summary())

    def load_config(self, run_id: str) -> dict:
        """
        Load configuration for a run.

        Args:
            run_id: The run ID.

        Returns:
            Configuration dictionary.
        """
        config_path = self.base_dir / run_id / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_metrics(self, run_id: str) -> dict:
        """
        Load metrics for a run.

        Args:
            run_id: The run ID.

        Returns:
            Metrics dictionary.
        """
        metrics_path = self.base_dir / run_id / "metrics.json"
        with open(metrics_path, "r") as f:
            return json.load(f)

    def load_samples(self, run_id: str, limit: Optional[int] = None) -> List[dict]:
        """
        Load samples for a run.

        Args:
            run_id: The run ID.
            limit: Maximum number of samples to load.

        Returns:
            List of sample dictionaries.
        """
        samples_path = self.base_dir / run_id / "samples.jsonl"
        samples = []

        with open(samples_path, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                samples.append(json.loads(line))

        return samples

    def load_run(self, run_id: str) -> dict:
        """
        Load all data for a run.

        Args:
            run_id: The run ID.

        Returns:
            Dictionary with config, metrics, and sample count.
        """
        run_dir = self.base_dir / run_id

        result = {
            "run_id": run_id,
            "config": None,
            "metrics": None,
            "sample_count": 0,
        }

        # Load config
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            result["config"] = self.load_config(run_id)

        # Load metrics
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            result["metrics"] = self.load_metrics(run_id)

        # Count samples
        samples_path = run_dir / "samples.jsonl"
        if samples_path.exists():
            with open(samples_path, "r") as f:
                result["sample_count"] = sum(1 for _ in f)

        return result

    def list_runs(self) -> List[str]:
        """
        List all available runs.

        Returns:
            List of run IDs, sorted by timestamp (newest first).
        """
        runs = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith("run_"):
                runs.append(path.name)

        # Sort by timestamp (part after "run_")
        runs.sort(reverse=True)
        return runs

    def list_runs_with_info(self) -> List[dict]:
        """
        List all runs with basic info.

        Returns:
            List of dictionaries with run info.
        """
        runs = []
        for run_id in self.list_runs():
            try:
                run_dir = self.base_dir / run_id

                info = {"run_id": run_id}

                # Try to load config for model info
                config_path = run_dir / "config.yaml"
                if config_path.exists():
                    config = self.load_config(run_id)
                    info["model"] = config.get("model", {}).get("path", "unknown")
                    info["timestamp"] = config.get("timestamp", "")

                # Try to load metrics for summary
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    metrics = self.load_metrics(run_id)
                    info["valid_rate"] = metrics.get("valid_rate", 0)
                    info["total_samples"] = metrics.get("total_samples", metrics.get("num_samples", 0))
                    # Benchmark-specific metrics
                    if "best_r2" in metrics:
                        info["best_r2"] = metrics.get("best_r2")
                    if "benchmark_name" in metrics:
                        info["benchmark"] = metrics.get("benchmark_name")

                runs.append(info)
            except Exception as e:
                logger.warning(f"Error loading run {run_id}: {e}")
                runs.append({"run_id": run_id, "error": str(e)})

        return runs

    def delete_run(self, run_id: str):
        """
        Delete a run and all its data.

        Args:
            run_id: The run ID to delete.
        """
        import shutil

        run_dir = self.base_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            logger.info(f"Deleted run: {run_id}")
        else:
            logger.warning(f"Run not found: {run_id}")

    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return (self.base_dir / run_id).exists()


def create_storage(base_dir: str = "results") -> ResultStorage:
    """
    Convenience function to create a storage instance.

    Args:
        base_dir: Base directory for results.

    Returns:
        ResultStorage instance.
    """
    return ResultStorage(base_dir)
