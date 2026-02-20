"""
Expression generation utilities.

Generates mathematical expressions using fine-tuned language models.
Supports configurable generation parameters and batch generation.
"""

import json
import logging
from typing import Optional, List
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for expression generation."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 100
    num_return_sequences: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "num_return_sequences": self.num_return_sequences,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass
class PromptConfig:
    """Configuration for prompt construction."""

    vars: List[str] = field(default_factory=lambda: ["x_1"])
    ops: List[str] = field(default_factory=lambda: ["+", "-", "*", "/", "sin", "cos"])
    cons: str = "C"
    format: str = "infix"  # 'infix' or 'prefix'

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "vars": self.vars,
            "ops": self.ops,
            "cons": self.cons,
            "format": self.format,
        }


class ExpressionGenerator:
    """Generates expressions using a language model."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the generator.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            config: Generation configuration. If None, use defaults.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.device = next(model.parameters()).device

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def build_prompt(self, prompt_config: Optional[PromptConfig] = None) -> str:
        """
        Build a prompt for expression generation.

        Args:
            prompt_config: Configuration for the prompt.

        Returns:
            Prompt string in JSON format.
        """
        if prompt_config is None:
            prompt_config = PromptConfig()

        prompt = {
            "vars": prompt_config.vars,
            "ops": prompt_config.ops,
            "cons": prompt_config.cons,
            "expr": "",  # Will be completed by model
        }

        # Build JSON prompt up to "expr": "
        prompt_str = json.dumps(prompt, separators=(",", ": "))[:-2]  # Remove closing "}
        return prompt_str

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Generate expressions from a prompt.

        Args:
            prompt: The prompt string.
            config: Optional config override.

        Returns:
            List of generated output strings.
        """
        cfg = config or self.config

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                num_return_sequences=cfg.num_return_sequences,
                do_sample=cfg.do_sample,
                repetition_penalty=cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode outputs
        generated = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated.append(text)

        return generated

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        batch_size: int = 8,
    ) -> List[List[str]]:
        """
        Generate expressions for multiple prompts in batches.

        Args:
            prompts: List of prompt strings.
            config: Optional config override.
            batch_size: Number of prompts to process at once.

        Returns:
            List of lists of generated outputs (one list per prompt).
        """
        cfg = config or self.config
        all_results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            # Tokenize batch with padding
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    num_return_sequences=cfg.num_return_sequences,
                    do_sample=cfg.do_sample,
                    repetition_penalty=cfg.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode outputs
            if cfg.num_return_sequences == 1:
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    all_results.append([text])
            else:
                # Handle multiple sequences per prompt
                for j in range(len(batch)):
                    start_idx = j * cfg.num_return_sequences
                    end_idx = start_idx + cfg.num_return_sequences
                    sequences = []
                    for output in outputs[start_idx:end_idx]:
                        text = self.tokenizer.decode(output, skip_special_tokens=True)
                        sequences.append(text)
                    all_results.append(sequences)

        return all_results

    def generate_n_samples(
        self,
        n: int,
        prompt_config: Optional[PromptConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        show_progress: bool = True,
    ) -> List[dict]:
        """
        Generate n expression samples.

        Args:
            n: Number of samples to generate.
            prompt_config: Configuration for prompts.
            generation_config: Configuration for generation.
            show_progress: Whether to show progress bar.

        Returns:
            List of dictionaries with prompt and output for each sample.
        """
        prompt = self.build_prompt(prompt_config)
        cfg = generation_config or self.config
        results = []

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(range(n), desc="Generating")
            except ImportError:
                iterator = range(n)
                logger.warning("tqdm not installed, progress bar disabled")
        else:
            iterator = range(n)

        for i in iterator:
            outputs = self.generate(prompt, cfg)
            results.append({
                "idx": i,
                "prompt": prompt,
                "outputs": outputs,
            })

        return results


def create_generator(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 100,
) -> ExpressionGenerator:
    """
    Convenience function to create a generator.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        ExpressionGenerator instance.
    """
    config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    return ExpressionGenerator(model, tokenizer, config)
