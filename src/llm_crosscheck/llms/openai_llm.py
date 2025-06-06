from collections.abc import AsyncIterator, Coroutine
from datetime import datetime
from typing import Any
from uuid import uuid4

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
        super().__init__(config)

        if config.provider != LLMProvider.OPENAI:
            raise LLMValidationError(
                f"Invalid provider {config.provider} for OpenAI LLM"
            )

        if not config.api_key:
            raise LLMValidationError("OpenAI API key is required")

        self._client: AsyncOpenAI = self._initialise_client()

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def supported_models(self) -> list[str]:
        return self.SUPPORTED_MODELS

    def _initialise_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
            base_url=self.config.base_url,
            organization=self.config.organisation,
        )

    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        try:
            messages = self._convert_messages_to_openai(request.messages)

            params: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": request.stream,
            }

            if request.functions:
                params["functions"] = request.functions
            if request.function_call:
                params["function_call"] = request.function_call

            params = {k: v for k, v in params.items() if v is not None}

            response = await self._client.chat.completions.create(**params)
            return self._convert_openai_response(response, request)

        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(
                str(e), provider=self.provider_name, model=request.model
            )
        except openai.RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    retry_after = float(retry_after_header)
            raise LLMRateLimitError(
                str(e),
                retry_after=retry_after,
                provider=self.provider_name,
                model=request.model,
            )
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise LLMConnectionError(
                str(e), provider=self.provider_name, model=request.model
            )
        except openai.BadRequestError as e:
            raise LLMValidationError(
                str(e), provider=self.provider_name, model=request.model
            )
        except openai.OpenAIError as e:
            raise LLMError(str(e), provider=self.provider_name, model=request.model)
        except Exception as e:
            raise LLMError(str(e), provider=self.provider_name, model=request.model)

    async def _stream_request(
        self, request: LLMRequest
    ) -> Coroutine[Any, Any, AsyncIterator[str]]:
        async def stream() -> AsyncIterator[str]:
            try:
                messages = self._convert_messages_to_openai(request.messages)

                params: dict[str, Any] = {
                    "model": request.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "frequency_penalty": request.frequency_penalty,
                    "presence_penalty": request.presence_penalty,
                    "stream": True,
                }

                params = {k: v for k, v in params.items() if v is not None}

                async for chunk in await self._client.chat.completions.create(**params):
                    if chunk.choices and chunk.choices[0].delta:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

            except Exception as e:
                raise LLMError(
                    f"OpenAI streaming error: {str(e)}",
                    provider=self.provider_name,
                    model=request.model,
                )

        return stream()

    def _convert_messages_to_openai(
        self, messages: list[LLMMessage]
    ) -> list[dict[str, Any]]:
        result = []
        for message in messages:
            msg: dict[str, Any] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.name:
                msg["name"] = message.name
            if message.function_call:
                msg["function_call"] = message.function_call
            result.append(msg)
        return result

    def _convert_openai_response(
        self, response: Any, request: LLMRequest
    ) -> LLMResponse:
        choices = []
        for choice in response.choices:
            msg = choice.message
            message = LLMMessage(
                role=LLMRole(msg.role),
                content=msg.content or "",
                function_call=getattr(msg, "function_call", None),
            )
            choices.append(
                LLMChoice(
                    index=choice.index,
                    message=message,
                    finish_reason=choice.finish_reason,
                )
            )

        usage = None
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            id=response.id,
            object=response.object,
            created=datetime.fromtimestamp(response.created),
            model=response.model,
            provider=LLMProvider.OPENAI,
            choices=choices,
            usage=usage,
            request_id=request.request_id,
            response_time_ms=None,  # Required to fix the missing argument error
        )
