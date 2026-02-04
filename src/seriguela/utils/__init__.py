"""
Seriguela - Classes for symbolic regression with LLMs.

This module provides core classes for expression parsing, validation,
and dataset management.
"""

from .expression import Expression
from .dataset import Dataset

__all__ = [
    'Expression',
    'Dataset',
]

__version__ = '1.0.0'
