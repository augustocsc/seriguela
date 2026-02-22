"""
RL algorithms for symbolic regression.

Provides:
- PurePPO: Standard PPO (no buffer)
- PureGRPO: Standard GRPO (no buffer)
- BoN-PPO: Best-of-N + PPO hybrid (with elite buffer)
- BoN-GRPO: Best-of-N + GRPO hybrid (with elite buffer)
- BestOfN: Pure sampling baseline (no RL training)
"""

import sys
from pathlib import Path

# Add paths for imports
REINFORCEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REINFORCEMENT_ROOT))

from algorithms.base_trainer import BaseRLTrainer, TrainerConfig
from algorithms.bon_ppo import BoNPPOTrainer
from algorithms.bon_grpo import BoNGRPOTrainer
from algorithms.pure_ppo import PurePPOTrainer
from algorithms.pure_grpo import PureGRPOTrainer
from algorithms.best_of_n import BestOfNBaseline, BoNConfig, run_best_of_n_baseline

__all__ = [
    "BaseRLTrainer",
    "TrainerConfig",
    "BoNPPOTrainer",
    "BoNGRPOTrainer",
    "PurePPOTrainer",
    "PureGRPOTrainer",
    "BestOfNBaseline",
    "BoNConfig",
    "run_best_of_n_baseline",
]
