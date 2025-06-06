"""
MongoDB ODM models for LLM request/response storage.
"""

from uuid import UUID

from beanie import Indexed
from pydantic import Field

from ..schemas.llm import LLMChoice, LLMMessage, LLMProvider, LLMUsage
from .base import TimestampedDocument


class LLMRequestDocument(TimestampedDocument):
    """MongoDB document for LLM requests."""

    # Request identification
    request_id: UUID = Field()
    user_id: str | None = Indexed(str, optional=True)
    session_id: str | None = Indexed(str, optional=True)

    # Request details
    provider: LLMProvider
    model: str
    messages: list[LLMMessage]

    # Generation parameters
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # Metadata
    stream: bool = False
    functions: list[dict] | None = None

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
    choices: list[LLMChoice]
    usage: LLMUsage | None = None

    # Performance metrics
    response_time_ms: float | None = None

    # Status
    status: str = "completed"  # completed, failed, timeout
    error_message: str | None = None

    class Settings:
        """Beanie settings."""

        name = "llm_responses"
        use_enum_values = True
        validate_on_save = True
