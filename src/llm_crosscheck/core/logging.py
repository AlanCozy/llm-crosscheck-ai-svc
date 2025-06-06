"""
Structured logging configuration for the LLM CrossCheck service.

This module provides centralized logging functionality with proper formatting,
context tracking, and integration with monitoring systems.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    service_name: str = "llm-crosscheck-ai-svc",
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs in JSON format
        service_name: Name of the service for log identification
    """
    # Clear existing handlers
    logging.root.handlers.clear()

    # Configure structlog
    structlog.configure(
        processors=[
            # Add service name and timestamp
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Convert to stdlib
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(handler)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def setup_logging(level: str = "info", log_format: str = "json") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
            if log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


# Initialize default logging configuration
configure_logging()
