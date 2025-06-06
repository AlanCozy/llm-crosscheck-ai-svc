"""
OpenAI LLM provider implementation.

This module provides integration with OpenAI's GPT models through their official API,
including support for both standard and streaming completions.
"""

from datetime import datetime
from typing import Any, AsyncIterator, Dict, List

import openai
from openai import AsyncOpenAI

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


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider implementation."""
    
    # OpenAI model configurations
    SUPPORTED_MODELS = [
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
    ]
    
    def __init__(self, config: LLMProviderConfig):
        """
        Initialise OpenAI LLM provider.
        
        Args:
            config: OpenAI-specific configuration
        """
        super().__init__(config)
        
        # Validate OpenAI-specific configuration
        if config.provider != LLMProvider.OPENAI:
            raise LLMValidationError(f"Invalid provider {config.provider} for OpenAI LLM")
        
        if not config.api_key:
            raise LLMValidationError("OpenAI API key is required")
    
    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return "OpenAI"
    
    @property
    def supported_models(self) -> List[str]:
        """Return list of models supported by this provider."""
        return self.SUPPORTED_MODELS
    
    async def _initialise_client(self) -> AsyncOpenAI:
        """
        Initialise the OpenAI client.
        
        Returns:
            The initialised AsyncOpenAI client
        """
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
        }
        
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        
        if self.config.organisation:
            client_kwargs["organization"] = self.config.organisation
        
        return AsyncOpenAI(**client_kwargs)
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request to OpenAI API.
        
        Args:
            request: Standardised LLM request
            
        Returns:
            Standardised LLM response
            
        Raises:
            LLMError: For any OpenAI-specific errors
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model,
                "messages": openai_messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": request.stream,
            }
            
            # Add functions if provided
            if request.functions:
                params["functions"] = request.functions
            
            if request.function_call:
                params["function_call"] = request.function_call
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Make the API call
            response = await self._client.chat.completions.create(**params)
            
            # Convert response to standardised format
            return self._convert_openai_response(response, request)
            
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(
                f"OpenAI authentication failed: {str(e)}",
                provider=self.provider_name,
                model=request.model
            )
        
        except openai.RateLimitError as e:
            # Extract retry-after header if available
            retry_after = None
            if hasattr(e, 'response') and e.response:
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    retry_after = float(retry_after)
            
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded: {str(e)}",
                retry_after=retry_after,
                provider=self.provider_name,
                model=request.model
            )
        
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise LLMConnectionError(
                f"OpenAI connection error: {str(e)}",
                provider=self.provider_name,
                model=request.model
            )
        
        except openai.BadRequestError as e:
            raise LLMValidationError(
                f"OpenAI request validation failed: {str(e)}",
                provider=self.provider_name,
                model=request.model
            )
        
        except openai.OpenAIError as e:
            raise LLMError(
                f"OpenAI API error: {str(e)}",
                provider=self.provider_name,
                model=request.model
            )
        
        except Exception as e:
            raise LLMError(
                f"Unexpected OpenAI error: {str(e)}",
                provider=self.provider_name,
                model=request.model
            )
    
    async def _stream_request(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Make a streaming request to OpenAI API.
        
        Args:
            request: Standardised LLM request
            
        Yields:
            String chunks from the streaming response
            
        Raises:
            LLMError: For any OpenAI-specific errors
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model,
                "messages": openai_messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": True,
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Make the streaming API call
            async for chunk in await self._client.chat.completions.create(**params):
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                        
        except Exception as e:
            # Reuse error handling from _make_request
            raise LLMError(
                f"OpenAI streaming error: {str(e)}",
                provider=self.provider_name,
                model=request.model
            )
    
    def _convert_messages_to_openai(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """
        Convert standardised messages to OpenAI format.
        
        Args:
            messages: List of standardised LLM messages
            
        Returns:
            List of OpenAI-formatted messages
        """
        openai_messages = []
        
        for message in messages:
            openai_message = {
                "role": message.role.value,
                "content": message.content,
            }
            
            if message.name:
                openai_message["name"] = message.name
            
            if message.function_call:
                openai_message["function_call"] = message.function_call
            
            openai_messages.append(openai_message)
        
        return openai_messages
    
    def _convert_openai_response(self, response: Any, request: LLMRequest) -> LLMResponse:
        """
        Convert OpenAI response to standardised format.
        
        Args:
            response: OpenAI API response
            request: Original request for correlation
            
        Returns:
            Standardised LLM response
        """
        # Convert choices
        choices = []
        for choice in response.choices:
            message = LLMMessage(
                role=LLMRole(choice.message.role),
                content=choice.message.content or "",
                function_call=getattr(choice.message, 'function_call', None)
            )
            
            llm_choice = LLMChoice(
                index=choice.index,
                message=message,
                finish_reason=choice.finish_reason
            )
            choices.append(llm_choice)
        
        # Convert usage information
        usage = None
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        return LLMResponse(
            id=response.id,
            object=response.object,
            created=datetime.fromtimestamp(response.created),
            model=response.model,
            provider=LLMProvider.OPENAI,
            choices=choices,
            usage=usage,
            request_id=request.request_id
        ) 