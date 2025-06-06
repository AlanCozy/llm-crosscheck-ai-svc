from collections.abc import AsyncIterator, Coroutine
from datetime import datetime
from typing import Any, cast
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
    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)

        if config.provider != LLMProvider.ANTHROPIC:
            raise LLMValidationError(
                f"Invalid provider {config.provider} for Anthropic LLM"
            )

        if not config.api_key:
            raise LLMValidationError("Anthropic API key is required")

        self._client: AsyncAnthropic = self._initialise_client()

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    @property
    def supported_models(self) -> list[str]:
        return self.SUPPORTED_MODELS

    def _initialise_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
            base_url=self.config.base_url,  # type: ignore[arg-type] if mypy complains
        )

    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        try:
            system_message, messages = self._convert_messages_to_anthropic(
                request.messages
            )

            params: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 1000,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream,
            }
            if system_message:
                params["system"] = system_message

            params = {k: v for k, v in params.items() if v is not None}

            response = await self._client.messages.create(**params)
            return self._convert_anthropic_response(response, request)

        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(
                str(e), provider=self.provider_name, model=request.model
            )
        except anthropic.RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after_header = e.response.headers.get("retry-after-seconds")
                if retry_after_header:
                    retry_after = float(retry_after_header)
            raise LLMRateLimitError(
                str(e),
                retry_after=retry_after,
                provider=self.provider_name,
                model=request.model,
            )
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise LLMConnectionError(
                str(e), provider=self.provider_name, model=request.model
            )
        except anthropic.BadRequestError as e:
            raise LLMValidationError(
                str(e), provider=self.provider_name, model=request.model
            )
        except anthropic.AnthropicError as e:
            raise LLMError(str(e), provider=self.provider_name, model=request.model)
        except Exception as e:
            raise LLMError(str(e), provider=self.provider_name, model=request.model)

    async def _stream_request(
        self, request: LLMRequest
    ) -> Coroutine[Any, Any, AsyncIterator[str]]:
        async def stream() -> AsyncIterator[str]:
            try:
                system_message, messages = self._convert_messages_to_anthropic(
                    request.messages
                )

                params: dict[str, Any] = {
                    "model": request.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens or 1000,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "stream": True,
                }
                if system_message:
                    params["system"] = system_message

                params = {k: v for k, v in params.items() if v is not None}

                async with self._client.messages.stream(**params) as stream:
                    async for text in stream.text_stream:
                        yield text
            except Exception as e:
                raise LLMError(str(e), provider=self.provider_name, model=request.model)

        return stream()

    def _convert_messages_to_anthropic(
        self, messages: list[LLMMessage]
    ) -> tuple[str, list[dict[str, str]]]:
        system_message = ""
        anthropic_messages = []

        for message in messages:
            if message.role == LLMRole.SYSTEM:
                system_message += (
                    f"\n\n{message.content}" if system_message else message.content
                )
            elif message.role in [LLMRole.USER, LLMRole.ASSISTANT]:
                anthropic_messages.append(
                    {
                        "role": message.role.value,
                        "content": message.content,
                    }
                )

        return system_message, anthropic_messages

    def _convert_anthropic_response(
        self, response: Any, request: LLMRequest
    ) -> LLMResponse:
        content = ""
        if response.content and len(response.content) > 0:
            content_block = response.content[0]
            content = getattr(content_block, "text", str(content_block))

        message = LLMMessage(role=LLMRole.ASSISTANT, content=content)
        choice = LLMChoice(
            index=0,
            message=message,
            finish_reason=getattr(response, "stop_reason", None),
        )

        usage = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            usage = LLMUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
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
            response_time_ms=None,  # fix: add this field if required by the LLMResponse class
        )
