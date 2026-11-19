"""
BoN-GRPO: Best-of-N + Group Relative Policy Optimization hybrid.

Combines Best-of-N sampling with GRPO training:
1. Maintains an elite buffer of best expressions
2. Uses group-relative advantages (within-batch ranking)
3. No explicit value function needed
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

# Add paths for imports
REINFORCEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REINFORCEMENT_ROOT))

from algorithms.base_trainer import BaseRLTrainer, TrainerConfig, Rollout
from rewards import BaseReward, PenaltyHandler
from schedulers import TemperatureScheduler
from callbacks import EarlyStoppingCallback
from buffers import EliteBuffer

logger = logging.getLogger(__name__)


class BoNGRPOTrainer(BaseRLTrainer):
    """
    Best-of-N GRPO trainer for symbolic regression.

    GRPO (Group Relative Policy Optimization):
    - Groups samples and computes relative advantages within groups
    - Uses ranking-based advantages instead of absolute rewards
    - More robust to reward scale and distribution

    Key features:
    1. Elite buffer maintains top-K expressions by R²
    2. Training batches mix new generations with buffer samples
    3. Group-relative advantage computation
    4. KL divergence constraint to prevent policy collapse
    """

    def __init__(
        self,
        config: TrainerConfig,
        x: np.ndarray,
        y: np.ndarray,
        reward_fn: BaseReward,
        penalty_handler: PenaltyHandler,
        temp_scheduler: TemperatureScheduler,
        early_stopping: EarlyStoppingCallback,
        elite_buffer: Optional[EliteBuffer] = None,
        is_prefix: bool = False,
        valid_variables: Optional[Set[str]] = None,
        ground_truth: Optional[str] = None,
    ):
        # Create buffer if not provided
        if elite_buffer is None:
            elite_buffer = EliteBuffer(
                max_size=config.buffer_size,
                sample_ratio=config.buffer_sample_ratio,
            )

        super().__init__(
            config=config,
            x=x,
            y=y,
            reward_fn=reward_fn,
            penalty_handler=penalty_handler,
            temp_scheduler=temp_scheduler,
            early_stopping=early_stopping,
            elite_buffer=elite_buffer,
            is_prefix=is_prefix,
            valid_variables=valid_variables,
            ground_truth=ground_truth,
        )

        self.group_size = config.group_size
        logger.info(f"BoN-GRPO initialized with group size {self.group_size}")

    def compute_advantages(self, rollouts: List[Rollout]) -> List[float]:
        """
        Compute group-relative advantages.

        GRPO advantage:
        For each group of samples, rank by reward and assign advantages
        based on position in ranking.

        A_i = (rank_i - mean_rank) / std_rank

        This makes the algorithm robust to reward scale.
        """
        import random
        random.shuffle(rollouts)

        n_samples = len(rollouts)
        advantages = np.zeros(n_samples)

        # Get rewards
        rewards = np.array([
            r.reward_result.reward if r.reward_result else -1.0
            for r in rollouts
        ])

        # Split into groups
        n_groups = max(1, n_samples // self.group_size)

        for g in range(n_groups):
            start_idx = g * self.group_size
            end_idx = min(start_idx + self.group_size, n_samples)

            if end_idx <= start_idx:
                continue

            group_rewards = rewards[start_idx:end_idx]

            # Compute ranks within group (higher reward = higher rank)
            ranks = np.argsort(np.argsort(group_rewards)).astype(float)

            # Normalize ranks to advantages
            rank_mean = np.mean(ranks)
            rank_std = np.std(ranks)

            if rank_std > 1e-8:
                group_advantages = (ranks - rank_mean) / rank_std
            else:
                group_advantages = ranks - rank_mean

            advantages[start_idx:end_idx] = group_advantages

        # Store advantages in rollouts
        for r, adv in zip(rollouts, advantages):
            r.advantage = adv

        return advantages.tolist()

    def update_policy(self, rollouts: List[Rollout], advantages: List[float]) -> Dict:
        """
        Perform GRPO policy update.

        GRPO uses a simpler update than PPO:
        L = -sum(advantage_i * log π(a_i|s)) / N

        With KL constraint:
        L_total = L_policy + β * KL(π_new || π_old)
        """
        self.model.train()

        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl = 0.0
        valid_count = 0

        # Filter valid rollouts with tokens
        valid_rollouts = [
            (r, adv) for r, adv in zip(rollouts, advantages)
            if r.reward_result and r.reward_result.is_valid and len(r.tokens) > 0
        ]

        if not valid_rollouts:
            return {
                "policy_loss": 0.0,
                "entropy_loss": 0.0,
                "kl_divergence": 0.0,
            }

        self.optimizer.zero_grad()

        for rollout, advantage in valid_rollouts:
            tokens = rollout.tokens
            old_log_probs = rollout.log_probs

            if len(tokens) == 0:
                continue

            # Compute new log probs
            full_ids = torch.cat([
                self.prompt_ids,
                torch.tensor([tokens], device=self.device)
            ], dim=1)

            temperature = self.temp_scheduler.get_temperature(
                self.current_step, self.config.max_steps
            )

            outputs = self.model(full_ids[:, :-1])
            logits = outputs.logits / temperature

            prompt_len = self.prompt_ids.shape[1]
            gen_logits = logits[:, prompt_len-1:, :]

            new_log_probs_all = F.log_softmax(gen_logits, dim=-1)
            new_probs_all = F.softmax(gen_logits, dim=-1)

            target_tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)
            new_log_probs_selected = new_log_probs_all.gather(
                2, target_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # Policy gradient loss (REINFORCE-style with advantage)
            advantage_tensor = torch.tensor(advantage, device=self.device)
            policy_loss = -(new_log_probs_selected.mean() * advantage_tensor)

            # KL divergence term
            old_log_probs_tensor = torch.tensor(
                old_log_probs, device=self.device
            ).unsqueeze(0)
            log_ratio = new_log_probs_selected - old_log_probs_tensor
            kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()

            # Entropy bonus
            entropy_per_token = -(new_probs_all * new_log_probs_all).sum(dim=-1)
            entropy_loss = -entropy_per_token.mean()

            # Combined loss with KL penalty
            loss = (
                policy_loss +
                self.config.max_kl * kl +  # KL coefficient as penalty
                self.config.entropy_coef * entropy_loss
            )
            loss = loss / len(valid_rollouts)
            loss.backward()

            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_kl += kl.item()
            valid_count += 1

        if valid_count > 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )
            self.optimizer.step()

        return {
            "policy_loss": total_policy_loss / max(valid_count, 1),
            "entropy_loss": total_entropy_loss / max(valid_count, 1),
            "kl_divergence": total_kl / max(valid_count, 1),
        }
