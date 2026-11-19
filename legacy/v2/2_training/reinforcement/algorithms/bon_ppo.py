"""
BoN-PPO: Best-of-N + Proximal Policy Optimization hybrid.

Combines Best-of-N sampling with PPO training:
1. Maintains an elite buffer of best expressions
2. Mixes buffer samples with new generations
3. Uses PPO clipped objective for policy updates
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


class BoNPPOTrainer(BaseRLTrainer):
    """
    Best-of-N PPO trainer for symbolic regression.

    Key features:
    1. Elite buffer maintains top-K expressions by R²
    2. Training batches mix new generations (80%) with buffer samples (20%)
    3. PPO clipped surrogate objective for stable updates
    4. Multiple PPO epochs per batch
    5. KL divergence monitoring for early stopping within epochs
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

        logger.info(f"BoN-PPO initialized with buffer size {config.buffer_size}")

    def compute_advantages(self, rollouts: List[Rollout]) -> List[float]:
        """
        Compute advantages using EMA baseline.

        Advantage = reward - baseline

        The baseline is updated as exponential moving average of rewards.
        """
        valid_rewards = [
            r.reward_result.reward
            for r in rollouts
            if r.reward_result and r.reward_result.is_valid
        ]

        if valid_rewards:
            mean_reward = np.mean(valid_rewards)
            self.baseline = (
                self.baseline_decay * self.baseline +
                (1 - self.baseline_decay) * mean_reward
            )

        advantages = []
        for r in rollouts:
            if r.reward_result and r.reward_result.is_valid:
                adv = r.reward_result.reward - self.baseline
            else:
                adv = -0.3  # Small fixed penalty for invalid
            advantages.append(adv)

        # Normalize advantages
        adv_array = np.array(advantages)
        adv_mean = np.mean(adv_array)
        adv_std = np.std(adv_array)
        if adv_std > 1e-8:
            advantages = ((adv_array - adv_mean) / adv_std).tolist()

        # Store advantages in rollouts
        for r, adv in zip(rollouts, advantages):
            r.advantage = adv

        return advantages

    def update_policy(self, rollouts: List[Rollout], advantages: List[float]) -> Dict:
        """
        Perform PPO update with multiple epochs.

        PPO objective:
        L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) + c_entropy * entropy

        Where:
        - ratio = π_new(a|s) / π_old(a|s)
        - A = advantage
        - ε = clip_epsilon
        """
        self.model.train()

        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        early_stopped = False

        # Filter valid rollouts with tokens (buffer samples won't have tokens)
        valid_indices = [
            i for i, r in enumerate(rollouts)
            if r.reward_result and r.reward_result.is_valid and len(r.tokens) > 0
        ]

        if not valid_indices:
            return {
                "policy_loss": 0.0,
                "entropy_loss": 0.0,
                "kl_divergence": 0.0,
                "ppo_early_stopped": False,
                "ppo_epochs_used": 0,
            }

        # PPO optimization epochs
        for ppo_epoch in range(self.config.ppo_epochs):
            epoch_kl = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            valid_count = 0

            self.optimizer.zero_grad()

            for idx in valid_indices:
                rollout = rollouts[idx]
                advantage = advantages[idx]

                tokens = rollout.tokens
                old_log_probs = rollout.log_probs

                if len(tokens) == 0:
                    continue

                # Compute new log probs under current policy
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

                # Compute ratio
                old_log_probs_tensor = torch.tensor(
                    old_log_probs, device=self.device
                ).unsqueeze(0)

                log_ratio = new_log_probs_selected - old_log_probs_tensor
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence
                kl = (ratio - 1 - log_ratio).mean()
                epoch_kl += kl.item()

                # PPO clipped objective
                advantage_tensor = torch.tensor(advantage, device=self.device)

                surr1 = ratio * advantage_tensor
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                )
                surr2 = clipped_ratio * advantage_tensor

                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_per_token = -(new_probs_all * new_log_probs_all).sum(dim=-1)
                entropy_loss = -entropy_per_token.mean()

                # Combined loss
                loss = policy_loss + self.config.entropy_coef * entropy_loss
                loss = loss / len(valid_indices)
                loss.backward()

                epoch_policy_loss += policy_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                valid_count += 1

            if valid_count > 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm
                )
                self.optimizer.step()

                avg_kl = epoch_kl / valid_count
                total_kl += avg_kl
                total_policy_loss += epoch_policy_loss / valid_count
                total_entropy_loss += epoch_entropy_loss / valid_count
                num_updates += 1

                # Early stopping if KL is too large
                if avg_kl > self.config.max_kl:
                    early_stopped = True
                    break

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "entropy_loss": total_entropy_loss / max(num_updates, 1),
            "kl_divergence": total_kl / max(num_updates, 1),
            "ppo_early_stopped": early_stopped,
            "ppo_epochs_used": num_updates,
            "baseline": self.baseline,
        }
