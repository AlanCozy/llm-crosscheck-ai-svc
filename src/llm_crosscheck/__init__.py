"""
LLM CrossCheck AI Service

A robust Python service for cross-checking and validating Large Language Model outputs.
"""

__version__ = "0.1.0"
__author__ = "Your Team"
__email__ = "team@yourcompany.com"
__description__ = "A robust Python service for cross-checking and validating Large Language Model outputs"

from .config.settings import get_settings
from .core.exceptions import LLMCrossCheckException

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "get_settings",
    "LLMCrossCheckException",
]
