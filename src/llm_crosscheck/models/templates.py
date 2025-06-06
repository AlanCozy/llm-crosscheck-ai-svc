"""
MongoDB ODM models for prompt template management.
"""

from typing import Dict, List, Optional

from beanie import Indexed
from pydantic import Field

from .base import TimestampedDocument


class PromptTemplateDocument(TimestampedDocument):
    """MongoDB document for prompt templates."""

    # Template identification
    name: str = Indexed(str, unique=True)
    version: str = "1.0.0"

    # Template content
    description: Optional[str] = None
    template: str
    category: Optional[str] = Indexed(str, optional=True)
    tags: List[str] = Field(default_factory=list)

    # Template validation
    required_variables: List[str] = Field(default_factory=list)
    optional_variables: List[str] = Field(default_factory=list)
    compiled_template: Optional[str] = None  # Cached compiled template

    # LLM configuration defaults
    default_model: Optional[str] = None
    default_max_tokens: Optional[int] = None
    default_temperature: Optional[float] = None

    # Usage tracking
    usage_count: int = 0
    last_used_at: Optional[str] = None

    # Metadata
    created_by: Optional[str] = None
    is_active: bool = True

    class Settings:
        """Beanie settings."""

        name = "prompt_templates"
        use_enum_values = True
        validate_on_save = True

    async def increment_usage(self) -> None:
        """Increment usage count and update last used time."""
        from datetime import datetime

        self.usage_count += 1
        self.last_used_at = datetime.utcnow().isoformat()
        await self.save_with_timestamp()
