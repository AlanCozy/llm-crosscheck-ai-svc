"""
MongoDB ODM models for prompt template management.
"""


from beanie import Indexed
from pydantic import Field

from .base import TimestampedDocument


class PromptTemplateDocument(TimestampedDocument):
    """MongoDB document for prompt templates."""

    # Template identification
    name: str = Indexed(str, unique=True)
    version: str = "1.0.0"

    # Template content
    description: str | None = None
    template: str
    category: str | None = Indexed(str, optional=True)
    tags: list[str] = Field(default_factory=list)

    # Template validation
    required_variables: list[str] = Field(default_factory=list)
    optional_variables: list[str] = Field(default_factory=list)
    compiled_template: str | None = None  # Cached compiled template

    # LLM configuration defaults
    default_model: str | None = None
    default_max_tokens: int | None = None
    default_temperature: float | None = None

    # Usage tracking
    usage_count: int = 0
    last_used_at: str | None = None

    # Metadata
    created_by: str | None = None
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
