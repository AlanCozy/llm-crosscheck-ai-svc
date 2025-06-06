"""
MongoDB ODM models for LLM request/response storage.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from beanie import Indexed
from pydantic import Field

from ..schemas.llm import LLMProvider, LLMMessage, LLMUsage, LLMChoice
from .base import TimestampedDocument


class LLMRequestDocument(TimestampedDocument):
    """MongoDB document for LLM requests."""
    
    # Request identification
    request_id: UUID = Field(unique=True)
    user_id: Optional[str] = Indexed(str, optional=True)
    session_id: Optional[str] = Indexed(str, optional=True)
    
    # Request details
    provider: LLMProvider
    model: str
    messages: List[LLMMessage]
    
    # Generation parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    # Metadata
    stream: bool = False
    functions: Optional[List[Dict]] = None
    
    class Settings:
        """Beanie settings."""
        name = "llm_requests"
        use_enum_values = True
        validate_on_save = True


class LLMResponseDocument(TimestampedDocument):
    """MongoDB document for LLM responses."""
    
    # Response identification
    response_id: str
    request_id: UUID = Indexed(UUID)
    
    # Response details
    provider: LLMProvider
    model: str
    object: str = "chat.completion"
    choices: List[LLMChoice]
    usage: Optional[LLMUsage] = None
    
    # Performance metrics
    response_time_ms: Optional[float] = None
    
    # Status
    status: str = "completed"  # completed, failed, timeout
    error_message: Optional[str] = None
    
    class Settings:
        """Beanie settings."""
        name = "llm_responses"
        use_enum_values = True
        validate_on_save = True 