"""
LLM schemas for request/response handling and configuration.

This module defines the core data schemas for interacting with Large Language Models,
providing standardised interfaces across different providers.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    MISTRAL = "mistral"
    LLAMA = "llama"
    HUGGING_FACE = "hugging_face"
    COHERE = "cohere"
    GOOGLE = "google"
    LOCAL = "local"


class LLMRole(str, Enum):
    """Message roles in LLM conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class LLMMessage(BaseModel):
    """Individual message in an LLM conversation."""

    role: LLMRole
    content: str
    name: str | None = None
    function_call: dict[str, Any] | None = None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class LLMRequest(BaseModel):
    """Standardised LLM request schema."""

    # Core request fields
    messages: list[LLMMessage]
    model: str

    # Generation parameters
    max_tokens: int | None = Field(default=1000, ge=1, le=32000)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)

    # Streaming and functions
    stream: bool = Field(default=False)
    functions: list[dict[str, Any]] | None = None
    function_call: str | dict[str, Any] | None = None

    # Request metadata
    request_id: UUID = Field(default_factory=uuid4)
    user_id: str | None = None
    session_id: str | None = None

    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within acceptable range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class LLMUsage(BaseModel):
    """Token usage information from LLM response."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @validator("total_tokens")
    def validate_total_tokens(cls, v: int, values: dict[str, Any]) -> int:
        """Ensure total tokens equals prompt + completion."""
        prompt_tokens = values.get("prompt_tokens", 0)
        completion_tokens = values.get("completion_tokens", 0)
        expected_total = prompt_tokens + completion_tokens

        if v != expected_total:
            # Log warning but don't fail - some providers might calculate differently
            pass

        return v


class LLMChoice(BaseModel):
    """Individual choice in LLM response."""

    index: int = Field(ge=0)
    message: LLMMessage
    finish_reason: str | None = None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class LLMResponse(BaseModel):
    """Standardised LLM response schema."""

    # Response identification
    id: str
    object: str = "chat.completion"
    created: datetime = Field(default_factory=datetime.utcnow)
    model: str
    provider: LLMProvider

    # Response content
    choices: list[LLMChoice]
    usage: LLMUsage | None = None

    # Request correlation
    request_id: UUID

    # Performance metrics
    response_time_ms: float | None = Field(ge=0)

    @property
    def content(self) -> str:
        """Get the content from the first choice."""
        if self.choices:
            return self.choices[0].message.content
        return ""

    @property
    def finish_reason(self) -> str | None:
        """Get the finish reason from the first choice."""
        if self.choices:
            return self.choices[0].finish_reason
        return None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class LLMProviderConfig(BaseModel):
    """Configuration schema for LLM providers."""

    provider: LLMProvider
    api_key: str
    base_url: str | None = None
    organisation: str | None = None

    # Rate limiting
    max_requests_per_minute: int = Field(default=60, ge=1)
    max_tokens_per_minute: int = Field(default=150000, ge=1)

    # Retry configuration
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)

    # Timeout configuration
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    # Model defaults
    default_model: str
    available_models: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class TemplateContext(BaseModel):
    """Context for template rendering."""

    variables: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from the context."""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a variable in the context."""
        self.variables[key] = value

    def update(self, **kwargs: Any) -> None:
        """Update multiple variables in the context."""
        self.variables.update(kwargs)


class PromptTemplate(BaseModel):
    """Prompt template configuration schema."""

    name: str = Field(min_length=1, max_length=100)
    description: str | None = None
    template: str = Field(min_length=1)

    # Template metadata
    version: str = Field(default="1.0.0")
    category: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Validation
    required_variables: list[str] = Field(default_factory=list)
    optional_variables: list[str] = Field(default_factory=list)

    # LLM configuration defaults
    default_model: str | None = None
    default_max_tokens: int | None = Field(default=None, ge=1, le=32000)
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    created_by: str | None = None

    @validator("template")
    def validate_template_syntax(cls, v: str) -> str:
        """Basic validation of Jinja2 template syntax."""
        try:
            from jinja2 import Template

            Template(v)
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template syntax: {e}")
        return v
