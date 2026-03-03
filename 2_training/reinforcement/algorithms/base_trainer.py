"""
Base trainer class for RL algorithms in symbolic regression.

Provides common functionality for:
- Model loading and management
- Prompt building and expression extraction
- Rollout collection
- Logging and checkpointing
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
REINFORCEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))
sys.path.insert(0, str(REINFORCEMENT_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

from classes.expression import Expression

# Import our new components (using absolute imports via sys.path)
from rewards import BaseReward, PenaltyHandler, PenaltyStrategy, RewardResult
from schedulers import TemperatureScheduler, FixedTemperature
from callbacks import EarlyStoppingCallback, EarlyStoppingConfig, StopReason
from buffers import EliteBuffer, BufferEntry

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for RL trainers."""
    # Model
    model_path: str = ""
    base_model: str = "gpt2"  # gpt2, gpt2-medium, gpt2-large

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 64
    max_steps: int = 10000
    max_new_tokens: int = 50

    # PPO specific
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    max_kl: float = 0.1
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # GRPO specific
    group_size: int = 8

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Buffer (for BoN-RL)
    buffer_size: int = 1000
    buffer_sample_ratio: float = 0.2

    # Early stopping
    patience: int = 5
    delta: float = 0.01
    r2_threshold: float = 0.999
    entropy_threshold: float = 0.1

    # Prompt type (standard, oracle, distractor)
    prompt_type: str = "standard"

    # Logging
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "./output"

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "seriguela"
    wandb_run_name: Optional[str] = None


@dataclass
class Rollout:
    """A single rollout (generated expression)."""
    text: str
    expression: str
    tokens: List[int]
    log_probs: List[float]
    total_log_prob: float
    reward_result: Optional[RewardResult] = None
    advantage: float = 0.0


class BaseRLTrainer(ABC):
    """
    Base class for RL trainers.

    Subclasses must implement:
    - compute_advantages(): How to compute advantages from rollouts
    - update_policy(): How to update policy given rollouts and advantages
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
        """
        Initialize trainer.

        Args:
            config: Trainer configuration
            x: Input data (n_samples, n_features)
            y: Target values (n_samples,)
            reward_fn: Reward function to use
            penalty_handler: Penalty handler for invalid expressions
            temp_scheduler: Temperature scheduler
            early_stopping: Early stopping callback
            elite_buffer: Elite buffer for BoN-RL (optional)
            is_prefix: Whether to use prefix notation
            valid_variables: Set of valid variable names
            ground_truth: Ground truth expression (for exact recovery)
        """
        self.config = config
        self.x = x
        self.y = y
        self.n_vars = x.shape[1]
        self.reward_fn = reward_fn
        self.penalty_handler = penalty_handler
        self.temp_scheduler = temp_scheduler
        self.early_stopping = early_stopping
        self.elite_buffer = elite_buffer
        self.is_prefix = is_prefix
        self.valid_variables = valid_variables or {f"x_{i+1}" for i in range(self.n_vars)}
        self.ground_truth = ground_truth

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model()

        # Build prompt
        self.prompt = self._build_prompt()
        self.prompt_ids = self.tokenizer(
            self.prompt, return_tensors="pt"
        )["input_ids"].to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            eps=1e-5,
        )

        # Tracking
        self.best_r2 = -np.inf
        self.best_expression = None
        self.best_reward = -np.inf
        self.history = []
        self.discovered_expressions: Dict[str, float] = {}
        self.current_step = 0

        # EMA baseline for advantage estimation
        self.baseline = 0.0
        self.baseline_decay = 0.95

        # Wandb
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()

    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try loading as LoRA adapter first
        try:
            logger.info("Attempting to load as LoRA adapter...")
            base_model = AutoModelForCausalLM.from_pretrained(self.config.base_model)

            if len(self.tokenizer) != base_model.config.vocab_size:
                base_model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Resized embeddings to {len(self.tokenizer)}")

            model_with_lora = PeftModel.from_pretrained(base_model, self.config.model_path)
            self.model = model_with_lora.merge_and_unload()
            logger.info("LoRA adapter loaded and merged successfully")

        except Exception as e:
            logger.info(f"LoRA load failed ({e}), loading as standalone model...")
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)

        # Add LoRA for training
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["c_attn"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model = self.model.to(self.device)

        # Use FP16 on CUDA for ~2x speedup (T4 has dedicated FP16 tensor cores)
        if self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Model converted to FP16 for faster inference")

        self.model.train()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model loaded with {trainable:,} trainable params")

    def _build_prompt(self, ops: Optional[List[str]] = None) -> str:
        """Build JSON format prompt based on prompt type."""
        vars_list = [f"x_{i+1}" for i in range(self.n_vars)]

        # Handle different prompt types
        if self.config.prompt_type in ["oracle", "distractor"]:
            from utils.prompt_builder import create_prompt_builder
            builder = create_prompt_builder(
                prompt_type=self.config.prompt_type,
                valid_variables=self.valid_variables,
                ground_truth=self.ground_truth,
            )
            return builder.build_prompt()

        # Standard prompt
        if ops is None:
            ops_list = ["+", "-", "*", "/", "sin", "cos", "sqrt", "log", "exp", "**"]
        else:
            ops_list = ops

        prompt = json.dumps({
            "vars": vars_list,
            "ops": ops_list,
            "cons": "C",
            "expr": ""
        })
        # Remove closing brace and quote to let model complete
        prompt = prompt[:-2]
        return prompt

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            run_name = self.config.wandb_run_name or (
                f"seriguela-{self.__class__.__name__}-"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "algorithm": self.__class__.__name__,
                    "reward_fn": self.reward_fn.name,
                    "penalty_strategy": self.penalty_handler.strategy.value,
                    "temp_scheduler": self.temp_scheduler.name,
                    "is_prefix": self.is_prefix,
                    **self.config.__dict__,
                },
            )
            logger.info(f"WandB initialized: {run_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None

    def extract_expression(self, text: str) -> str:
        """Extract expression from generated text."""
        try:
            eos_token = "<|endoftext|>"
            if eos_token in text:
                text = text[:text.index(eos_token)]

            if '"expr": "' in text:
                start = text.index('"expr": "') + len('"expr": "')
                remaining = text[start:]
                for terminator in ['"}', '"']:
                    if terminator in remaining:
                        return remaining[:remaining.index(terminator)].strip()
                return remaining.strip()

            if '"expr": ' in text:
                start = text.index('"expr": ') + len('"expr": ')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                return remaining.strip(' "')

        except (ValueError, IndexError):
            pass

        if '"expr"' in text:
            return text.split('"expr"')[-1].strip(' ":{}')
        return text.strip()

    def generate_expression(self, temperature: float) -> Rollout:
        """Generate a single expression."""
        generated_ids = self.prompt_ids.clone()
        generated_tokens = []
        log_probs_list = []

        with torch.no_grad():
            for step in range(self.config.max_new_tokens):
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                token_log_prob = log_probs[0, next_token.item()].item()

                generated_tokens.append(next_token.item())
                log_probs_list.append(token_log_prob)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check for end of generation
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                if '"}' in text[len(self.prompt):]:
                    break

        # Decode
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        expr_str = self.extract_expression(text)

        return Rollout(
            text=text,
            expression=expr_str,
            tokens=generated_tokens,
            log_probs=log_probs_list,
            total_log_prob=sum(log_probs_list),
        )

    # Max sequences per GPU pass — 512 uses ~3GB on T4 with FP16
    GPU_BATCH_SIZE = 512

    def _generate_sub_batch(self, num_samples: int, temperature: float) -> List[Rollout]:
        """Generate a sub-batch of rollouts on GPU. Called by collect_rollouts."""
        batch_prompt_ids = self.prompt_ids.expand(num_samples, -1)
        prompt_len = self.prompt_ids.shape[1]

        with torch.no_grad():
            # output_scores=True gives us per-step logits for free
            # return_dict_in_generate=True combined with use_cache=True is critical
            # Without use_cache=True, auto-regressive generation is O(N^2) instead of O(N)
            gen_output = self.model.generate(
                batch_prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True, # Critical for generation speed
            )

        sequences = gen_output.sequences  # (batch, prompt_len + gen_len)
        scores = gen_output.scores         # tuple of (batch, vocab_size) per step

        # Vectorized Log-Prob Extraction (Fully on GPU)
        # scores is a tuple of length `gen_len`, each item is a tensor of shape (batch, vocab_size)
        # We stack it along axis 1 to get (batch, gen_len, vocab_size)
        all_logits = torch.stack(scores, dim=1)
        
        # We need the generated tokens to gather their log_probs.
        # sequences is (batch, prompt_len + max_gen_len)
        generated_tokens_tensor = sequences[:, prompt_len:] # (batch, max_gen_len)

        # Truncate all_logits to match the actual generated length (in case generation stopped early)
        actual_gen_len = generated_tokens_tensor.shape[1]
        all_logits = all_logits[:, :actual_gen_len, :]

        # Compute log softmax across vocabulary
        all_log_probs = F.log_softmax(all_logits, dim=-1) # (batch, gen_len, vocab_size)

        # Gather the log_probs of the specific tokens we generated
        # We use unsqueeze to make generated_tokens_tensor (batch, gen_len, 1) to match all_log_probs
        gathered_log_probs = torch.gather(all_log_probs, 2, generated_tokens_tensor.unsqueeze(-1)).squeeze(-1) # (batch, gen_len)
        
        # Move back to CPU for list parsing
        sequences_cpu = sequences.cpu()
        gathered_log_probs_cpu = gathered_log_probs.cpu()

        # Build rollouts
        rollouts = []
        for i in range(num_samples):
            seq = sequences_cpu[i]
            generated_tokens = seq[prompt_len:].tolist()

            # Remove padding (EOS tokens at the end)
            eos_id = self.tokenizer.eos_token_id
            clean_tokens = []
            for t in generated_tokens:
                clean_tokens.append(t)
                if t == eos_id:
                    break
            
            # Extract only the log probs for the clean tokens
            clean_len = len(clean_tokens)
            log_probs_list = gathered_log_probs_cpu[i, :clean_len].tolist()

            # Decode text and extract expression
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            expr_str = self.extract_expression(text)

            rollouts.append(Rollout(
                text=text,
                expression=expr_str,
                tokens=clean_tokens,
                log_probs=log_probs_list,
                total_log_prob=sum(log_probs_list),
            ))

        return rollouts

    def collect_rollouts(self, num_samples: int) -> List[Rollout]:
        """Collect rollouts using sub-batched GPU generation.

        Generates in chunks of GPU_BATCH_SIZE to avoid OOM, then evaluates
        rewards on CPU. The total num_samples can be much larger than what
        fits in GPU memory at once.
        """
        self.model.eval()

        temperature = self.temp_scheduler.get_temperature(
            self.current_step, self.config.max_steps
        )

        # 1. Generate in GPU-friendly sub-batches
        rollouts = []
        remaining = num_samples
        while remaining > 0:
            chunk = min(remaining, self.GPU_BATCH_SIZE)
            rollouts.extend(self._generate_sub_batch(chunk, temperature))
            remaining -= chunk

        # 2. Compute rewards on CPU
        for rollout in rollouts:
            reward_result = self.penalty_handler.compute_with_penalty(
                self.reward_fn,
                rollout.expression,
                self.x,
                self.y,
                self.is_prefix
            )
            rollout.reward_result = reward_result

            if reward_result.is_valid:
                self.discovered_expressions[rollout.expression] = max(
                    self.discovered_expressions.get(rollout.expression, -np.inf),
                    reward_result.r2
                )

            if reward_result.r2 > self.best_r2:
                self.best_r2 = reward_result.r2
                self.best_expression = rollout.expression

            if reward_result.reward > self.best_reward:
                self.best_reward = reward_result.reward

        return rollouts


    def compute_policy_entropy(self) -> float:
        """Compute current policy entropy (single forward pass)."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.prompt_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()
        return entropy

    @abstractmethod
    def compute_advantages(self, rollouts: List[Rollout]) -> List[float]:
        """Compute advantages for rollouts. Must be implemented by subclass."""
        pass

    @abstractmethod
    def update_policy(self, rollouts: List[Rollout], advantages: List[float]) -> Dict:
        """Update policy given rollouts and advantages. Must be implemented by subclass."""
        pass

    def train_step(self) -> Dict:
        """Perform one training step."""
        # Collect rollouts
        rollouts = self.collect_rollouts(self.config.batch_size)

        # Add to elite buffer if available
        if self.elite_buffer is not None:
            self.elite_buffer.add_batch(
                expressions=[r.expression for r in rollouts],
                r2_scores=[r.reward_result.r2 for r in rollouts],
                rewards=[r.reward_result.reward for r in rollouts],
                log_probs=[r.total_log_prob for r in rollouts],
                complexities=[r.reward_result.complexity for r in rollouts],
                current_step=self.current_step
            )

        # Sample from buffer and add to rollouts
        if self.elite_buffer is not None and len(self.elite_buffer) > 0:
            buffer_samples = self.elite_buffer.sample(self.config.batch_size)
            for entry in buffer_samples:
                # Create rollout from buffer entry (no tokens/log_probs)
                buffer_rollout = Rollout(
                    text="",
                    expression=entry.expression,
                    tokens=[],
                    log_probs=[],
                    total_log_prob=entry.log_prob,
                    reward_result=RewardResult(
                        reward=entry.reward,
                        r2=entry.r2,
                        mse=0.0,
                        is_valid=True,
                        complexity=entry.complexity,
                        error_type=None,
                        expression=entry.expression,
                    ),
                )
                rollouts.append(buffer_rollout)

        # Compute advantages
        advantages = self.compute_advantages(rollouts)

        # Update policy
        update_stats = self.update_policy(rollouts, advantages)

        # Separate fresh rollouts from buffer samples (fresh have tokens)
        fresh_rollouts = [r for r in rollouts if len(r.tokens) > 0]
        buffer_rollouts = [r for r in rollouts if len(r.tokens) == 0]

        # Compute statistics for ALL rollouts
        rewards = [r.reward_result.reward for r in rollouts if r.reward_result]
        r2_values = [r.reward_result.r2 for r in rollouts if r.reward_result]
        valid_mask = [r.reward_result.is_valid for r in rollouts if r.reward_result]
        valid_r2 = [r2 for r2, v in zip(r2_values, valid_mask) if v]

        # Compute statistics for FRESH generations only (key for measuring improvement!)
        fresh_rewards = [r.reward_result.reward for r in fresh_rollouts if r.reward_result]
        fresh_r2_values = [r.reward_result.r2 for r in fresh_rollouts if r.reward_result]
        fresh_valid_mask = [r.reward_result.is_valid for r in fresh_rollouts if r.reward_result]
        fresh_valid_r2 = [r2 for r2, v in zip(fresh_r2_values, fresh_valid_mask) if v]

        # Track when best was found
        if not hasattr(self, 'best_step'):
            self.best_step = 0
        current_max = max(r2_values) if r2_values else 0
        if current_max >= self.best_r2 and current_max > 0:
            self.best_step = self.current_step

        # Unique expressions this step (diversity measure)
        unique_exprs_this_step = len(set(r.expression for r in fresh_rollouts if r.expression))

        stats = {
            "step": self.current_step,
            # Overall stats (includes buffer)
            "valid_count": int(sum(valid_mask)),
            "total_count": len(rollouts),
            "valid_rate": sum(valid_mask) / len(rollouts) if rollouts else 0,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_r2": float(np.mean(valid_r2)) if valid_r2 else 0.0,
            "max_r2": float(np.max(r2_values)) if r2_values else 0.0,
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "best_step": self.best_step,
            "temperature": self.temp_scheduler.get_temperature(
                self.current_step, self.config.max_steps
            ),

            # FRESH generation stats (KEY for measuring policy improvement!)
            "fresh_count": len(fresh_rollouts),
            "fresh_valid_count": int(sum(fresh_valid_mask)) if fresh_valid_mask else 0,
            "fresh_valid_rate": sum(fresh_valid_mask) / len(fresh_rollouts) if fresh_rollouts else 0,
            "fresh_mean_r2": float(np.mean(fresh_valid_r2)) if fresh_valid_r2 else 0.0,
            "fresh_max_r2": float(np.max(fresh_r2_values)) if fresh_r2_values else 0.0,
            "fresh_mean_reward": float(np.mean(fresh_rewards)) if fresh_rewards else 0.0,

            # Distribution stats (for fresh generations)
            "fresh_median_r2": float(np.median(fresh_valid_r2)) if fresh_valid_r2 else 0.0,
            "fresh_std_r2": float(np.std(fresh_valid_r2)) if len(fresh_valid_r2) > 1 else 0.0,
            "fresh_p75_r2": float(np.percentile(fresh_valid_r2, 75)) if fresh_valid_r2 else 0.0,
            "fresh_p90_r2": float(np.percentile(fresh_valid_r2, 90)) if fresh_valid_r2 else 0.0,

            # Diversity stats
            "unique_expressions": unique_exprs_this_step,
            "total_unique_discovered": len(self.discovered_expressions),

            # Buffer stats
            "buffer_samples_used": len(buffer_rollouts),

            **update_stats,
        }

        if self.elite_buffer is not None:
            stats["buffer_stats"] = self.elite_buffer.stats()

        # Log to wandb
        if self.wandb_run is not None:
            import wandb
            wandb.log(stats, step=self.current_step)

        self.current_step += 1
        return stats

    def run(self) -> Dict:
        """Run training loop."""
        logger.info("=" * 60)
        logger.info(f"{self.__class__.__name__} TRAINING")
        logger.info("=" * 60)
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Reward function: {self.reward_fn.name}")
        logger.info(f"Penalty strategy: {self.penalty_handler.strategy.value}")
        logger.info(f"Temperature scheduler: {self.temp_scheduler.name}")
        logger.info("=" * 60)

        try:
            while self.current_step < self.config.max_steps:
                stats = self.train_step()
                self.history.append(stats)

                # Log periodically
                if self.current_step % self.config.log_every == 0:
                    logger.info(
                        f"Step {stats['step']:5d} | "
                        f"Fresh: {stats['fresh_valid_count']}/{stats['fresh_count']} ({stats['fresh_valid_rate']*100:.0f}%) | "
                        f"Fresh R²: {stats['fresh_mean_r2']:.4f} (p90: {stats['fresh_p90_r2']:.4f}) | "
                        f"Best: {self.best_r2:.4f}@{stats['best_step']} | "
                        f"T: {stats['temperature']:.2f}"
                    )

                # Check early stopping
                policy_entropy = self.compute_policy_entropy()
                stop_reason = self.early_stopping.check(
                    step=self.current_step,
                    mean_reward=stats["mean_reward"],
                    best_r2=self.best_r2,
                    best_expr=self.best_expression or "",
                    policy_entropy=policy_entropy
                )

                if stop_reason != StopReason.NONE:
                    logger.info(f"Early stopping: {stop_reason.value}")
                    break

                # Save checkpoint
                if self.current_step % self.config.save_every == 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        # Final results
        self._log_final_results()
        self.save_results()

        if self.wandb_run is not None:
            self.wandb_run.finish()

        return self._get_results()

    def _log_final_results(self):
        """Log final training results."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best R²: {self.best_r2:.4f}")
        logger.info(f"Best expression: {self.best_expression}")
        logger.info(f"Unique expressions: {len(self.discovered_expressions)}")

        top_exprs = sorted(
            self.discovered_expressions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        logger.info("Top 5 expressions:")
        for expr, r2 in top_exprs:
            logger.info(f"  R²={r2:.4f}: {expr}")

    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{self.current_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def save_results(self):
        """Save training results."""
        results = self._get_results()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"results_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved: {output_path}")

    def _get_results(self) -> Dict:
        """Get results dictionary."""
        return {
            "algorithm": self.__class__.__name__,
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "total_steps": self.current_step,
            "history": self.history,
            "discovered_expressions": dict(list(self.discovered_expressions.items())[:100]),
            "config": self.config.__dict__,
            "reward_fn": self.reward_fn.name,
            "penalty_strategy": self.penalty_handler.strategy.value,
            "temp_scheduler": self.temp_scheduler.name,
            "early_stopping_summary": self.early_stopping.get_summary(),
        }
