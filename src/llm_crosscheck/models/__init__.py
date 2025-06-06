"""
MongoDB ODM models for the LLM CrossCheck service.

This module contains Beanie (MongoDB ODM) document models
for persistent data storage.
"""

from .audit import AuditLogDocument
from .base import TimestampedDocument
from .llm_requests import LLMRequestDocument, LLMResponseDocument
from .templates import PromptTemplateDocument

__all__ = [
    # Base models
    "TimestampedDocument",
    # LLM models
    "LLMRequestDocument",
    "LLMResponseDocument",
    # Template models
    "PromptTemplateDocument",
    # Audit models
    "AuditLogDocument",
]
