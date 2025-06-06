"""
MongoDB ODM models for the LLM CrossCheck service.

This module contains Beanie (MongoDB ODM) document models
for persistent data storage.
"""

from .base import TimestampedDocument
from .llm_requests import LLMRequestDocument, LLMResponseDocument
from .templates import PromptTemplateDocument
from .audit import AuditLogDocument

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