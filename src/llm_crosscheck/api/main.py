"""
FastAPI application entry point for LLM CrossCheck AI Service.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from ..config.settings import get_settings
from ..exceptions.exceptions import LLMCrossCheckException
from ..core.logging import setup_logging
from ..core.middleware import (
    LoggingMiddleware,
    ProcessTimeMiddleware,
    RateLimitMiddleware,
)
from ..routers import auth, crosscheck, health, metrics

# Setup structured logging
logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for application lifespan events."""
    # Startup
    logger.info("Starting LLM CrossCheck AI Service", version=settings.APP_VERSION)
    setup_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
    
    # Initialize any connections, caches, etc.
    # await init_database()
    # await init_redis()
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM CrossCheck AI Service")
    # Clean up connections, etc.
    # await close_database()
    # await close_redis()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Add middleware
    setup_middleware(app)
    
    # Add routers
    setup_routers(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.ALLOWED_METHODS,
        allow_headers=settings.ALLOWED_HEADERS,
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(ProcessTimeMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    if settings.RATE_LIMIT_PER_MINUTE > 0:
        app.add_middleware(
            RateLimitMiddleware,
            rate_limit=settings.RATE_LIMIT_PER_MINUTE,
            burst=settings.RATE_LIMIT_BURST,
        )


def setup_routers(app: FastAPI) -> None:
    """Configure application routers."""
    
    # API routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(crosscheck.router, prefix="/api/v1/crosscheck", tags=["CrossCheck"])
    
    # Metrics endpoint for Prometheus
    if settings.ENABLE_METRICS:
        app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""
    
    @app.exception_handler(LLMCrossCheckException)
    async def llm_crosscheck_exception_handler(
        request: Request, exc: LLMCrossCheckException
    ) -> JSONResponse:
        """Handle custom LLM CrossCheck exceptions."""
        logger.error(
            "LLM CrossCheck exception occurred",
            error=str(exc),
            error_code=exc.error_code,
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": str(exc),
                "details": exc.details,
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions."""
        logger.error(
            "Unhandled exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
            method=request.method,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An internal server error occurred",
            },
        )


# Create the FastAPI application
app = create_app()


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint with service information."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs" if settings.DEBUG else "Documentation disabled in production",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.llm_crosscheck.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    ) 