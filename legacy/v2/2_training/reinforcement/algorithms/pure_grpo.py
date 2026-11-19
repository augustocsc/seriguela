"""
Pure GRPO: Group Relative Policy Optimization without elite buffer.

Unlike BoN-GRPO, this implementation:
1. Does NOT maintain an elite buffer
2. Uses only fresh generations for each training step
3. Provides a baseline for comparing hybrid approaches
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

logger = logging.getLogger(__name__)


class PureGRPOTrainer(BaseRLTrainer):
    """
    Pure GRPO trainer for symbolic regression (no elite buffer).

    GRPO (Group Relative Policy Optimization):
    - Groups samples and computes relative advantages within groups
    - Uses ranking-based advantages instead of absolute rewards
    - More robust to reward scale and distribution

    Key features:
    1. No elite buffer - uses only fresh generations
    2. Group-relative advantage computation
    3. KL divergence constraint to prevent policy collapse
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
        elite_buffer=None,  # Accepted but ignored for API compatibility
        is_prefix: bool = False,
        valid_variables: Optional[Set[str]] = None,
        ground_truth: Optional[str] = None,
    ):
        # Initialize WITHOUT elite buffer
        super().__init__(
            config=config,
            x=x,
            y=y,
            reward_fn=reward_fn,
            penalty_handler=penalty_handler,
            temp_scheduler=temp_scheduler,
            early_stopping=early_stopping,
            elite_buffer=None,  # Force no buffer
            is_prefix=is_prefix,
            valid_variables=valid_variables,
            ground_truth=ground_truth,
        )

        self.group_size = config.group_size
        logger.info(f"Pure GRPO initialized with group size {self.group_size} (no elite buffer)")

    def training_step(self) -> Dict:
        """
        Execute one GRPO training step using only fresh generations.

        Override base class to skip buffer sampling.
        """
        # Generate fresh samples ONLY
        rollouts = self.generate_rollouts(self.config.batch_size)

        # Compute rewards
        for rollout in rollouts:
            if rollout.expression:
                rollout.reward_result = self.compute_reward(rollout.expression)

        # Compute group-relative advantages
        advantages = self.compute_advantages(rollouts)

        # Update policy with GRPO
        update_info = self.update_policy(rollouts, advantages)

        # Collect metrics
        valid_rollouts = [
            r for r in rollouts
            if r.reward_result and r.reward_result.is_valid
        ]

        metrics = {
            "step": self.current_step,
            "n_samples": len(rollouts),
            "valid_rate": len(valid_rollouts) / len(rollouts) if rollouts else 0,
            "temperature": self.temp_scheduler.get_temperature(
                self.current_step, self.config.max_steps
            ),
        }

        if valid_rollouts:
            rewards = [r.reward_result.reward for r in valid_rollouts]
            r2_values = [r.reward_result.r2 for r in valid_rollouts if r.reward_result.r2 is not None]

            metrics["mean_reward"] = np.mean(rewards)
            metrics["max_reward"] = np.max(rewards)
            if r2_values:
                metrics["mean_r2"] = np.mean(r2_values)
                metrics["max_r2"] = np.max(r2_values)

                # Track best
                best_idx = np.argmax(r2_values)
                if r2_values[best_idx] > self.best_r2:
                    self.best_r2 = r2_values[best_idx]
                    self.best_expression = valid_rollouts[best_idx].expression

            metrics["best_r2"] = self.best_r2
            if self.best_expression:
                metrics["best_expression"] = self.best_expression

        metrics.update(update_info)
        return metrics

    def compute_advantages(self, rollouts: List[Rollout]) -> List[float]:
        """
        Compute group-relative advantages.

        GRPO advantage:
        For each group of samples, rank by reward and assign advantages
        based on position in ranking.

        A_i = (rank_i - mean_rank) / std_rank

        This makes the algorithm robust to reward scale.
        """
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
        L = -sum(advantage_i * log pi(a_i|s)) / N

        With KL constraint:
        L_total = L_policy + beta * KL(pi_new || pi_old)
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
