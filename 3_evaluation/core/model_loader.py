"""
Model loading utilities for LoRA adapters.

Supports loading models from:
- HuggingFace Hub (e.g., augustocsc/gpt2_large_infix_682k)
- Local paths (e.g., ./models/gpt2/large_infix)
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads LoRA models from HuggingFace or local paths."""

    # Mapping from model size keywords to base model names
    BASE_MODEL_MAP = {
        "gpt2": "gpt2",
        "gpt2-base": "gpt2",
        "base": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "large": "gpt2-large",
    }

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the model loader.

        Args:
            device: Device to load the model on. If None, auto-detect (cuda if available).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"ModelLoader initialized with device: {self.device}")

    def _detect_base_model(self, model_path: str) -> str:
        """
        Detect the base model from adapter_config.json or model path.

        Args:
            model_path: Path to the LoRA adapter (HF repo or local path).

        Returns:
            Base model name (e.g., 'gpt2-large').
        """
        # Try to read adapter_config.json
        config_paths = [
            Path(model_path) / "adapter_config.json",  # Local path
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    base_model = config.get("base_model_name_or_path", "")
                    if base_model:
                        logger.info(f"Detected base model from config: {base_model}")
                        return base_model
                except Exception as e:
                    logger.warning(f"Error reading adapter config: {e}")

        # Try to infer from model path name
        model_path_lower = model_path.lower()
        for keyword, base_model in self.BASE_MODEL_MAP.items():
            if keyword in model_path_lower:
                logger.info(f"Inferred base model from path: {base_model}")
                return base_model

        # Default to gpt2
        logger.warning("Could not detect base model, defaulting to 'gpt2'")
        return "gpt2"

    def _is_huggingface_repo(self, model_path: str) -> bool:
        """Check if the path looks like a HuggingFace repo."""
        # HF repos typically have format: username/model_name
        if "/" in model_path and not os.path.exists(model_path):
            return True
        return False

    def load(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        merge_adapter: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
        """
        Load a LoRA model from HuggingFace or local path.

        Args:
            model_path: Path to the LoRA adapter (HF repo or local).
            base_model: Base model name. If None, auto-detect.
            merge_adapter: Whether to merge adapter weights into base model.
            torch_dtype: Data type for model weights. If None, use float16 on GPU.

        Returns:
            Tuple of (model, tokenizer, base_model_name).
        """
        logger.info(f"Loading model from: {model_path}")

        # Detect base model if not provided
        if base_model is None:
            base_model = self._detect_base_model(model_path)
        logger.info(f"Using base model: {base_model}")

        # Set dtype
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load tokenizer from adapter repo (has special tokens)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load base model
        logger.info(f"Loading base model: {base_model}")
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model_instance, model_path)

        # Optionally merge adapter weights
        if merge_adapter:
            logger.info("Merging adapter weights...")
            model = model.merge_and_unload()

        # Move to device if not using device_map
        if self.device != "cuda" or "device_map" not in str(model.hf_device_map if hasattr(model, "hf_device_map") else ""):
            model = model.to(self.device)

        # Set to eval mode
        model.eval()

        logger.info(f"Model loaded successfully. Device: {self.device}")
        return model, tokenizer, base_model

    def get_model_info(self, model_path: str) -> dict:
        """
        Get information about a model without loading it.

        Args:
            model_path: Path to the model.

        Returns:
            Dictionary with model information.
        """
        info = {
            "path": model_path,
            "is_huggingface": self._is_huggingface_repo(model_path),
            "base_model": self._detect_base_model(model_path),
        }

        # Try to read adapter config for more details
        config_path = Path(model_path) / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                info["lora_r"] = config.get("r")
                info["lora_alpha"] = config.get("lora_alpha")
                info["target_modules"] = config.get("target_modules")
            except Exception:
                pass

        return info


def load_model(
    model_path: str,
    device: Optional[str] = None,
    base_model: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """
    Convenience function to load a model.

    Args:
        model_path: Path to the LoRA adapter.
        device: Device to load on.
        base_model: Base model name.

    Returns:
        Tuple of (model, tokenizer, base_model_name).
    """
    loader = ModelLoader(device=device)
    return loader.load(model_path, base_model=base_model)
