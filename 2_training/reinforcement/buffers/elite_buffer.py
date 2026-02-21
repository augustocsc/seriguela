"""
Elite buffer for Best-of-N RL methods.

Maintains a buffer of the best expressions found during training
to anchor policy updates and prevent catastrophic forgetting.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import heapq
import random
import numpy as np


@dataclass
class BufferEntry:
    """
    Entry in the elite buffer.

    Stores expression along with its metrics for ranking and sampling.
    """
    expression: str
    r2: float
    reward: float
    log_prob: float
    complexity: int = 0
    step_added: int = 0

    def __lt__(self, other: "BufferEntry") -> bool:
        """Comparison for min-heap (lowest R² at top)."""
        return self.r2 < other.r2

    def __eq__(self, other: "BufferEntry") -> bool:
        return self.expression == other.expression

    def __hash__(self) -> int:
        return hash(self.expression)

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "r2": self.r2,
            "reward": self.reward,
            "log_prob": self.log_prob,
            "complexity": self.complexity,
            "step_added": self.step_added,
        }


class EliteBuffer:
    """
    Buffer of elite (best) expressions for BoN-RL methods.

    The buffer maintains the top-K expressions by R² score using a min-heap.
    During training, expressions from the buffer are mixed with newly
    generated expressions to anchor policy updates.

    Configuration from experimental plan:
    - Buffer Size: 1000
    - Buffer Strategy: Top-K by R²
    - Refresh Rate: Every 100 steps
    - Sampling from Buffer: 20% of batch
    """

    def __init__(
        self,
        max_size: int = 1000,
        sample_ratio: float = 0.2,
        diversity_threshold: float = 0.0,
        deduplicate: bool = True
    ):
        """
        Initialize elite buffer.

        Args:
            max_size: Maximum number of expressions to store
            sample_ratio: Fraction of batch to sample from buffer (0.2 = 20%)
            diversity_threshold: Minimum R² difference for adding similar expressions
            deduplicate: Whether to prevent duplicate expressions
        """
        self.max_size = max_size
        self.sample_ratio = sample_ratio
        self.diversity_threshold = diversity_threshold
        self.deduplicate = deduplicate

        # Use min-heap to efficiently maintain top-K
        self._heap: List[BufferEntry] = []
        # Set for fast duplicate checking
        self._expressions: set = set()
        # Track statistics
        self._total_added = 0
        self._total_rejected = 0

    def add(self, entry: BufferEntry) -> bool:
        """
        Add entry to buffer if it qualifies.

        Entry is added if:
        1. Buffer not full, or
        2. Entry's R² > lowest R² in buffer

        Args:
            entry: BufferEntry to add

        Returns:
            True if entry was added, False otherwise
        """
        # Skip invalid entries
        if entry.r2 <= 0 or not np.isfinite(entry.r2):
            return False

        # Check for duplicates
        if self.deduplicate and entry.expression in self._expressions:
            return False

        # Add if buffer not full
        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, entry)
            self._expressions.add(entry.expression)
            self._total_added += 1
            return True

        # Replace if better than worst
        if entry.r2 > self._heap[0].r2:
            # Check diversity threshold
            if self.diversity_threshold > 0:
                if entry.r2 - self._heap[0].r2 < self.diversity_threshold:
                    self._total_rejected += 1
                    return False

            # Remove worst entry
            removed = heapq.heapreplace(self._heap, entry)
            self._expressions.discard(removed.expression)
            self._expressions.add(entry.expression)
            self._total_added += 1
            return True

        self._total_rejected += 1
        return False

    def add_batch(
        self,
        expressions: List[str],
        r2_scores: List[float],
        rewards: List[float],
        log_probs: List[float],
        complexities: Optional[List[int]] = None,
        current_step: int = 0
    ) -> int:
        """
        Add batch of expressions to buffer.

        Args:
            expressions: List of expression strings
            r2_scores: List of R² scores
            rewards: List of reward values
            log_probs: List of log probabilities
            complexities: List of complexity values (optional)
            current_step: Current training step

        Returns:
            Number of entries added
        """
        if complexities is None:
            complexities = [len(e.split()) for e in expressions]

        added = 0
        for i, (expr, r2, reward, lp, comp) in enumerate(
            zip(expressions, r2_scores, rewards, log_probs, complexities)
        ):
            entry = BufferEntry(
                expression=expr,
                r2=r2,
                reward=reward,
                log_prob=lp,
                complexity=comp,
                step_added=current_step
            )
            if self.add(entry):
                added += 1

        return added

    def sample(self, batch_size: int) -> List[BufferEntry]:
        """
        Sample entries from buffer for training batch.

        Args:
            batch_size: Total batch size (buffer provides sample_ratio * batch_size)

        Returns:
            List of sampled BufferEntry objects
        """
        n_from_buffer = int(batch_size * self.sample_ratio)

        if len(self._heap) == 0 or n_from_buffer == 0:
            return []

        # Sample without replacement if possible
        n_to_sample = min(n_from_buffer, len(self._heap))
        return random.sample(self._heap, n_to_sample)

    def sample_weighted(self, batch_size: int, temperature: float = 1.0) -> List[BufferEntry]:
        """
        Sample entries weighted by R² score.

        Higher R² expressions are more likely to be sampled.

        Args:
            batch_size: Total batch size
            temperature: Sampling temperature (higher = more uniform)

        Returns:
            List of sampled BufferEntry objects
        """
        n_from_buffer = int(batch_size * self.sample_ratio)

        if len(self._heap) == 0 or n_from_buffer == 0:
            return []

        # Compute weights from R² scores
        r2_values = np.array([e.r2 for e in self._heap])
        # Shift to positive and apply temperature
        weights = np.exp((r2_values - r2_values.max()) / max(temperature, 0.01))
        weights = weights / weights.sum()

        # Sample with replacement using weights
        indices = np.random.choice(
            len(self._heap),
            size=min(n_from_buffer, len(self._heap)),
            replace=False,
            p=weights
        )

        return [self._heap[i] for i in indices]

    def get_best(self, k: int = 10) -> List[BufferEntry]:
        """
        Get the K best expressions by R².

        Args:
            k: Number of top expressions to return

        Returns:
            List of top K BufferEntry objects, sorted by R² descending
        """
        return heapq.nlargest(k, self._heap, key=lambda x: x.r2)

    def get_worst(self) -> Optional[BufferEntry]:
        """Get the worst (lowest R²) entry in buffer."""
        if not self._heap:
            return None
        return self._heap[0]

    def clear(self):
        """Clear all entries from buffer."""
        self._heap.clear()
        self._expressions.clear()
        self._total_added = 0
        self._total_rejected = 0

    def stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        if not self._heap:
            return {
                "size": 0,
                "max_size": self.max_size,
                "fill_ratio": 0.0,
                "mean_r2": 0.0,
                "max_r2": 0.0,
                "min_r2": 0.0,
                "std_r2": 0.0,
                "mean_complexity": 0.0,
                "total_added": self._total_added,
                "total_rejected": self._total_rejected,
            }

        r2_values = [e.r2 for e in self._heap]
        complexities = [e.complexity for e in self._heap]

        return {
            "size": len(self._heap),
            "max_size": self.max_size,
            "fill_ratio": len(self._heap) / self.max_size,
            "mean_r2": float(np.mean(r2_values)),
            "max_r2": float(np.max(r2_values)),
            "min_r2": float(np.min(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_complexity": float(np.mean(complexities)),
            "total_added": self._total_added,
            "total_rejected": self._total_rejected,
        }

    def get_config(self) -> dict:
        """Get buffer configuration."""
        return {
            "max_size": self.max_size,
            "sample_ratio": self.sample_ratio,
            "diversity_threshold": self.diversity_threshold,
            "deduplicate": self.deduplicate,
        }

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"EliteBuffer(size={stats['size']}/{self.max_size}, "
            f"mean_r2={stats['mean_r2']:.4f}, max_r2={stats['max_r2']:.4f})"
        )
