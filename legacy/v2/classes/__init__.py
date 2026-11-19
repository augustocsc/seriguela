"""
Seriguela - Classes for symbolic regression with LLMs.

This module provides core classes for expression parsing, validation,
and dataset management.
"""

from .expression import Expression
from .dataset import RegressionDataset

__all__ = [
    'Expression',
    'RegressionDataset',
]

__version__ = '1.0.0'
