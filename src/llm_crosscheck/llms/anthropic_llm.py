"""
Anthropic LLM provider implementation.

This module provides integration with Anthropic's Claude models through their official API,
including support for both standard and streaming completions.
"""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from uuid import uuid4

import anthropic
from anthropic import AsyncAnthropic

from ..schemas.llm import (
    LLMChoice,
    LLMMessage,
    LLMProvider,
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMUsage,
)
from .base import (
    BaseLLM,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMValidationError,
)


class AnthropicLLM(BaseLLM):
    """Anthropic LLM provider implementation."""

    # Anthropic model configurations
    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    def __init__(self, config: LLMProviderConfig):
        """
        Initialise Anthropic LLM provider.

        Args:
            config: Anthropic-specific configuration
        """
        super().__init__(config)

        # Validate Anthropic-specific configuration
        if config.provider != LLMProvider.ANTHROPIC:
            raise LLMValidationError(
                f"Invalid provider {config.provider} for Anthropic LLM"
            )

        if not config.api_key:
            raise LLMValidationError("Anthropic API key is required")

    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return "Anthropic"

    @property
    def supported_models(self) -> list[str]:
        """Return list of models supported by this provider."""
        return self.SUPPORTED_MODELS

    async def _initialise_client(self) -> AsyncAnthropic:
        """
        Initialise the Anthropic client.

        Returns:
            The initialised AsyncAnthropic client
        """
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
        }

        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        return AsyncAnthropic(**client_kwargs)

    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request to Anthropic API.

        Args:
            request: Standardised LLM request

        Returns:
            Standardised LLM response

        Raises:
            LLMError: For any Anthropic-specific errors
        """
        try:
            # Convert messages to Anthropic format
            system_message, messages = self._convert_messages_to_anthropic(
                request.messages
            )

            # Prepare request parameters
            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens
                or 1000,  # Anthropic requires max_tokens
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream,
            }

            # Add system message if present
            if system_message:
                params["system"] = system_message

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Make the API call
            response = await self._client.messages.create(**params)

            # Convert response to standardised format
            return self._convert_anthropic_response(response, request)

        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(
                f"Anthropic authentication failed: {str(e)}",
                provider=self.provider_name,
                model=request.model,
            )

        except anthropic.RateLimitError as e:
            # Extract retry-after header if available
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after-seconds")
                if retry_after:
                    retry_after = float(retry_after)

            raise LLMRateLimitError(
                f"Anthropic rate limit exceeded: {str(e)}",
                retry_after=retry_after,
                provider=self.provider_name,
                model=request.model,
            )

        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise LLMConnectionError(
                f"Anthropic connection error: {str(e)}",
                provider=self.provider_name,
                model=request.model,
            )

        except anthropic.BadRequestError as e:
            raise LLMValidationError(
                f"Anthropic request validation failed: {str(e)}",
                provider=self.provider_name,
                model=request.model,
            )

        except anthropic.AnthropicError as e:
            raise LLMError(
                f"Anthropic API error: {str(e)}",
                provider=self.provider_name,
                model=request.model,
            )

        except Exception as e:
            raise LLMError(
                f"Unexpected Anthropic error: {str(e)}",
                provider=self.provider_name,
                model=request.model,
            )

    async def _stream_request(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Make a streaming request to Anthropic API.

        Args:
            request: Standardised LLM request

        Yields:
            String chunks from the streaming response

        Raises:
            LLMError: For any Anthropic-specific errors
        """
        try:
            # Convert messages to Anthropic format
            system_message, messages = self._convert_messages_to_anthropic(
                request.messages
            )

            # Prepare request parameters
            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 1000,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": True,
            }

            # Add system message if present
            if system_message:
                params["system"] = system_message

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Make the streaming API call
            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            # Reuse error handling from _make_request
            raise LLMError(
                f"Anthropic streaming error: {str(e)}",
                provider=self.provider_name,
                model=request.model,
            )

    def _convert_messages_to_anthropic(
        self, messages: list[LLMMessage]
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Convert standardised messages to Anthropic format.

        Anthropic uses a different format where system messages are separate,
        and only user/assistant roles are allowed in the messages array.

        Args:
            messages: List of standardised LLM messages

        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_message = ""
        anthropic_messages = []

        for message in messages:
            if message.role == LLMRole.SYSTEM:
                # Anthropic handles system messages separately
                if system_message:
                    system_message += "\n\n" + message.content
                else:
                    system_message = message.content
            elif message.role in [LLMRole.USER, LLMRole.ASSISTANT]:
                anthropic_message = {
                    "role": message.role.value,
                    "content": message.content,
                }
                anthropic_messages.append(anthropic_message)
            # Skip function messages as Anthropic doesn't support them in the same way

        return system_message, anthropic_messages

    def _convert_anthropic_response(
        self, response: Any, request: LLMRequest
    ) -> LLMResponse:
        """
        Convert Anthropic response to standardised format.

        Args:
            response: Anthropic API response
            request: Original request for correlation

        Returns:
            Standardised LLM response
        """
        # Extract content from Anthropic response
        content = ""
        if response.content and len(response.content) > 0:
            # Anthropic returns content as a list of content blocks
            content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )

        # Create standardised message
        message = LLMMessage(role=LLMRole.ASSISTANT, content=content)

        # Create choice
        choice = LLMChoice(
            index=0,
            message=message,
            finish_reason=getattr(response, "stop_reason", None),
        )

        # Convert usage information
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = LLMUsage(
                prompt_tokens=getattr(response.usage, "input_tokens", 0),
                completion_tokens=getattr(response.usage, "output_tokens", 0),
                total_tokens=getattr(response.usage, "input_tokens", 0)
                + getattr(response.usage, "output_tokens", 0),
            )

        return LLMResponse(
            id=getattr(response, "id", str(uuid4())),
            object="chat.completion",
            created=datetime.utcnow(),
            model=getattr(response, "model", request.model),
            provider=LLMProvider.ANTHROPIC,
            choices=[choice],
            usage=usage,
            request_id=request.request_id,
        )
