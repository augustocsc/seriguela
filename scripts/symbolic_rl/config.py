from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    model_name: str = "gpt2"
    prompt: str = "Find the best expression for the dataset:"
    dataset_path: str = "data/data.csv"
    stop_reward: float = 0.99  # critério de parada baseado em R²
    max_epochs: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-5
    generation_max_length: int = 64
    device: str = "cuda"
    output_dir: str = "checkpoints"
    log_interval: int = 10
    seed: int = 42
