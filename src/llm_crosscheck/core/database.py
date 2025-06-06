"""
MongoDB database configuration and connection management.
"""

import os
from typing import List, Optional

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure

from ..models import (
    AuditLogDocument,
    LLMRequestDocument,
    LLMResponseDocument,
    PromptTemplateDocument,
)
from .logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """MongoDB database connection manager."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        
    async def connect(self, mongodb_url: Optional[str] = None) -> None:
        """Connect to MongoDB database."""
        if not mongodb_url:
            mongodb_url = os.getenv(
                "MONGODB_URL", 
                "mongodb://admin:password@localhost:27017/llm_crosscheck?authSource=admin"
            )
        
        try:
            # Create MongoDB client
            self.client = AsyncIOMotorClient(mongodb_url)
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            # Get database name from URL or use default
            db_name = mongodb_url.split('/')[-1].split('?')[0] or "llm_crosscheck"
            self.database = self.client[db_name]
            
            # Initialize Beanie ODM
            await init_beanie(
                database=self.database,
                document_models=[
                    LLMRequestDocument,
                    LLMResponseDocument,
                    PromptTemplateDocument,
                    AuditLogDocument,
                ]
            )
            
            logger.info(f"Initialized Beanie ODM with database: {db_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def health_check(self) -> bool:
        """Check database connection health."""
        try:
            if not self.client:
                return False
            
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def create_indexes(self) -> None:
        """Create database indexes for performance."""
        try:
            # LLM Requests indexes
            await LLMRequestDocument.get_motor_collection().create_index([
                ("request_id", 1)
            ], unique=True)
            
            await LLMRequestDocument.get_motor_collection().create_index([
                ("user_id", 1),
                ("created_at", -1)
            ])
            
            # LLM Responses indexes
            await LLMResponseDocument.get_motor_collection().create_index([
                ("request_id", 1)
            ])
            
            await LLMResponseDocument.get_motor_collection().create_index([
                ("provider", 1),
                ("model", 1),
                ("created_at", -1)
            ])
            
            # Prompt Templates indexes
            await PromptTemplateDocument.get_motor_collection().create_index([
                ("name", 1)
            ], unique=True)
            
            await PromptTemplateDocument.get_motor_collection().create_index([
                ("category", 1),
                ("is_active", 1)
            ])
            
            # Audit Logs indexes
            await AuditLogDocument.get_motor_collection().create_index([
                ("event_type", 1),
                ("created_at", -1)
            ])
            
            await AuditLogDocument.get_motor_collection().create_index([
                ("user_id", 1),
                ("created_at", -1)
            ])
            
            logger.info("Successfully created database indexes")
            
        except Exception as e:
            logger.error(f"Failed to create database indexes: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


async def get_database():
    """Get database instance."""
    return db_manager.database


async def init_database(mongodb_url: Optional[str] = None) -> None:
    """Initialize database connection."""
    await db_manager.connect(mongodb_url)
    await db_manager.create_indexes()


async def close_database() -> None:
    """Close database connection."""
    await db_manager.disconnect() 