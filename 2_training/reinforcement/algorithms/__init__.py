"""
RL algorithms for symbolic regression.

Provides:
- PPO: Proximal Policy Optimization
- GRPO: Group Relative Policy Optimization
- BoN-PPO: Best-of-N + PPO hybrid
- BoN-GRPO: Best-of-N + GRPO hybrid
"""

import sys
from pathlib import Path

# Add paths for imports
REINFORCEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REINFORCEMENT_ROOT))

from algorithms.base_trainer import BaseRLTrainer, TrainerConfig
from algorithms.bon_ppo import BoNPPOTrainer
from algorithms.bon_grpo import BoNGRPOTrainer

__all__ = [
    "BaseRLTrainer",
    "TrainerConfig",
    "BoNPPOTrainer",
    "BoNGRPOTrainer",
]
