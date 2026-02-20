"""
Core library for model evaluation.

This module provides the common utilities for:
- Model loading (LoRA adapters from HuggingFace or local)
- Expression generation with configurable parameters
- Expression extraction from model outputs
- Expression validation using the Expression class
- Metrics calculation
- Result persistence
"""

from .model_loader import ModelLoader
from .generator import ExpressionGenerator, GenerationConfig
from .extractor import ExpressionExtractor
from .validator import ExpressionValidator, ValidationResult
from .metrics import MetricsCalculator, QualityMetrics
from .storage import ResultStorage
from .hf_storage import HFResultStorage

__all__ = [
    "ModelLoader",
    "ExpressionGenerator",
    "GenerationConfig",
    "ExpressionExtractor",
    "ExpressionValidator",
    "ValidationResult",
    "MetricsCalculator",
    "QualityMetrics",
    "ResultStorage",
    "HFResultStorage",
]
