"""
Base MongoDB ODM models for common functionality.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from beanie import Document
from pydantic import Field


class TimestampedDocument(Document):
    """Base document with timestamp fields."""
    
    id: UUID = Field(default_factory=uuid4, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    class Settings:
        """Beanie settings."""
        use_enum_values = True
        validate_on_save = True
    
    async def save_with_timestamp(self, **kwargs) -> "TimestampedDocument":
        """Save document with updated timestamp."""
        self.updated_at = datetime.utcnow()
        return await self.save(**kwargs) 