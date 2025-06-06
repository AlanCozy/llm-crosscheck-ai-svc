"""
MongoDB ODM models for audit logging.
"""

from typing import Any, Dict, Optional

from beanie import Indexed
from pydantic import Field

from .base import TimestampedDocument


class AuditLogDocument(TimestampedDocument):
    """MongoDB document for audit logs."""
    
    # Event identification
    event_id: str = Field(unique=True)
    event_type: str = Indexed(str)  # request, response, error, auth, etc.
    
    # Actor information
    user_id: Optional[str] = Indexed(str, optional=True)
    session_id: Optional[str] = Indexed(str, optional=True)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Event details
    action: str  # create, read, update, delete, execute
    resource: str  # llm_request, template, user, etc.
    resource_id: Optional[str] = None
    
    # Event data
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Status and error information
    status: str = "success"  # success, error, warning
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Performance
    duration_ms: Optional[float] = None
    
    class Settings:
        """Beanie settings."""
        name = "audit_logs"
        use_enum_values = True
        validate_on_save = True 