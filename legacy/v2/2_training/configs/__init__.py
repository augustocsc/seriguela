"""
Seriguela - Configuration utilities.

This module provides standardized configuration utilities for
experiment tracking and naming conventions.
"""

from .wandb_config import (
    generate_run_name,
    get_wandb_project_name,
    parse_run_name,
    is_valid_run_name,
)

__all__ = [
    'generate_run_name',
    'get_wandb_project_name',
    'parse_run_name',
    'is_valid_run_name',
]

__version__ = '1.0.0'
