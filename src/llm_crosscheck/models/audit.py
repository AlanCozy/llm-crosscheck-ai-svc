"""
MongoDB ODM models for audit logging.
"""

from typing import Any

from beanie import Indexed
from pydantic import Field

from .base import TimestampedDocument


class AuditLogDocument(TimestampedDocument):
    """MongoDB document for audit logs."""

    # Event identification
    event_id: str = Field(unique=True)
    event_type: str = Indexed(str)  # request, response, error, auth, etc.

    # Actor information
    user_id: str | None = Indexed(str, optional=True)
    session_id: str | None = Indexed(str, optional=True)
    ip_address: str | None = None
    user_agent: str | None = None

    # Event details
    action: str  # create, read, update, delete, execute
    resource: str  # llm_request, template, user, etc.
    resource_id: str | None = None

    # Event data
    request_data: dict[str, Any] | None = None
    response_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Status and error information
    status: str = "success"  # success, error, warning
    error_code: str | None = None
    error_message: str | None = None

    # Performance
    duration_ms: float | None = None

    class Settings:
        """Beanie settings."""

        name = "audit_logs"
        use_enum_values = True
        validate_on_save = True
