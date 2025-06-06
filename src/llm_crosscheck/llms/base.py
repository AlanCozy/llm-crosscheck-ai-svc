"""
Abstract base class for LLM providers.

This module defines the standard interface that all LLM providers must implement,
ensuring consistent behaviour across different AI service providers.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from ..models.llm import LLMRequest, LLMResponse, LLMProviderConfig
from ..core.logging import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None):
        self.message = message
        self.provider = provider
        self.model = model
        super().__init__(self.message)


class LLMConnectionError(LLMError):
    """Raised when there's a connection issue with the LLM provider."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMValidationError(LLMError):
    """Raised when input validation fails."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the standard interface that all LLM implementations
    must follow, ensuring consistent behaviour across different providers.
    """
    
    def __init__(self, config: LLMProviderConfig):
        """
        Initialise the LLM provider.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self._client = None
        self._request_count = 0
        self._token_count = 0
        self._last_request_time = 0.0
        
        logger.info(
            f"Initialising {config.provider.value} LLM provider",
            extra={
                "provider": config.provider.value,
                "model": config.default_model,
                "base_url": config.base_url,
            }
        )
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Return list of models supported by this provider."""
        pass
    
    @abstractmethod
    async def _initialise_client(self) -> Any:
        """
        Initialise the provider-specific client.
        
        Returns:
            The initialised client instance
        """
        pass
    
    @abstractmethod
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request to the LLM provider.
        
        Args:
            request: Standardised LLM request
            
        Returns:
            Standardised LLM response
            
        Raises:
            LLMError: For any provider-specific errors
        """
        pass
    
    @abstractmethod
    async def _stream_request(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Make a streaming request to the LLM provider.
        
        Args:
            request: Standardised LLM request
            
        Yields:
            String chunks from the streaming response
            
        Raises:
            LLMError: For any provider-specific errors
        """
        pass
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            request: Standardised LLM request
            
        Returns:
            Standardised LLM response
            
        Raises:
            LLMError: For any errors during generation
        """
        await self._ensure_initialised()
        await self._check_rate_limits(request)
        
        start_time = time.time()
        
        try:
            # Validate the request
            self._validate_request(request)
            
            # Make the request with retry logic
            response = await self._make_request_with_retry(request)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            response.response_time_ms = response_time_ms
            
            # Update counters
            self._update_counters(request, response)
            
            logger.info(
                f"Generated response from {self.provider_name}",
                extra={
                    "provider": self.provider_name,
                    "model": request.model,
                    "request_id": str(request.request_id),
                    "response_time_ms": response_time_ms,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                }
            )
            
            return response
            
        except Exception as e:
            error_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error generating response from {self.provider_name}",
                extra={
                    "provider": self.provider_name,
                    "model": request.model,
                    "request_id": str(request.request_id),
                    "error_time_ms": error_time_ms,
                    "error": str(e),
                }
            )
            
            if isinstance(e, LLMError):
                raise
            else:
                raise LLMError(f"Unexpected error: {str(e)}", self.provider_name, request.model)
    
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            request: Standardised LLM request
            
        Yields:
            String chunks from the streaming response
            
        Raises:
            LLMError: For any errors during generation
        """
        await self._ensure_initialised()
        await self._check_rate_limits(request)
        
        start_time = time.time()
        
        try:
            # Validate the request
            self._validate_request(request)
            
            # Stream the response
            async for chunk in self._stream_request(request):
                yield chunk
                
            # Update counters (estimated)
            self._update_stream_counters(request)
            
            response_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Streamed response from {self.provider_name}",
                extra={
                    "provider": self.provider_name,
                    "model": request.model,
                    "request_id": str(request.request_id),
                    "response_time_ms": response_time_ms,
                }
            )
            
        except Exception as e:
            error_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error streaming response from {self.provider_name}",
                extra={
                    "provider": self.provider_name,
                    "model": request.model,
                    "request_id": str(request.request_id),
                    "error_time_ms": error_time_ms,
                    "error": str(e),
                }
            )
            
            if isinstance(e, LLMError):
                raise
            else:
                raise LLMError(f"Unexpected error: {str(e)}", self.provider_name, request.model)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the LLM provider.
        
        Returns:
            Health status information
        """
        try:
            await self._ensure_initialised()
            
            # Basic connectivity test
            health_status = {
                "provider": self.provider_name,
                "status": "healthy",
                "models": self.supported_models,
                "config": {
                    "default_model": self.config.default_model,
                    "base_url": self.config.base_url,
                    "timeout": self.config.timeout_seconds,
                },
                "metrics": {
                    "total_requests": self._request_count,
                    "total_tokens": self._token_count,
                },
            }
            
            return health_status
            
        except Exception as e:
            logger.warning(
                f"Health check failed for {self.provider_name}",
                extra={"provider": self.provider_name, "error": str(e)}
            )
            
            return {
                "provider": self.provider_name,
                "status": "unhealthy",
                "error": str(e),
            }
    
    async def _ensure_initialised(self) -> None:
        """Ensure the client is initialised."""
        if self._client is None:
            self._client = await self._initialise_client()
    
    async def _make_request_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Make a request with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._make_request(request)
                
            except LLMRateLimitError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = e.retry_after or self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit for {self.provider_name}, retrying in {wait_time}s",
                        extra={
                            "provider": self.provider_name,
                            "attempt": attempt + 1,
                            "wait_time": wait_time,
                        }
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except (LLMConnectionError, LLMError) as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Request failed for {self.provider_name}, retrying in {wait_time}s",
                        extra={
                            "provider": self.provider_name,
                            "attempt": attempt + 1,
                            "wait_time": wait_time,
                            "error": str(e),
                        }
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        # This shouldn't be reached, but just in case
        if last_exception:
            raise last_exception
    
    def _validate_request(self, request: LLMRequest) -> None:
        """Validate the request before sending."""
        if not request.messages:
            raise LLMValidationError("Request must contain at least one message")
        
        if request.model not in self.supported_models:
            logger.warning(
                f"Model {request.model} not in supported models list",
                extra={
                    "provider": self.provider_name,
                    "model": request.model,
                    "supported_models": self.supported_models,
                }
            )
        
        if request.max_tokens and request.max_tokens > 32000:
            raise LLMValidationError("max_tokens cannot exceed 32000")
    
    async def _check_rate_limits(self, request: LLMRequest) -> None:
        """Check rate limits before making a request."""
        current_time = time.time()
        
        # Simple rate limiting logic
        if self._last_request_time > 0:
            time_since_last = current_time - self._last_request_time
            min_interval = 60.0 / self.config.max_requests_per_minute
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.info(
                    f"Rate limiting: waiting {wait_time:.2f}s before next request",
                    extra={"provider": self.provider_name, "wait_time": wait_time}
                )
                await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    def _update_counters(self, request: LLMRequest, response: LLMResponse) -> None:
        """Update internal counters."""
        self._request_count += 1
        if response.usage:
            self._token_count += response.usage.total_tokens
    
    def _update_stream_counters(self, request: LLMRequest) -> None:
        """Update counters for streaming requests (estimated)."""
        self._request_count += 1
        # For streaming, we can't easily track tokens, so we don't update token count 