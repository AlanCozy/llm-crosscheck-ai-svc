"""
LLM Manager service for orchestrating multiple LLM providers and prompt templates.
"""

from pathlib import Path
from typing import Any, cast

from ..core.logging import LoggerMixin, get_logger
from ..core.prompt_engine import PromptEngine
from ..llms.factory import LLMFactory
from ..schemas.llm import (
    LLMMessage,
    LLMProvider,
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
    LLMRole,
    TemplateContext,
)

logger = get_logger(__name__)


class LLMManager(LoggerMixin):
    def __init__(
        self,
        prompt_dirs: list[str] | None = None,
        default_provider: LLMProvider = LLMProvider.OPENAI,
    ):
        if prompt_dirs is None:
            project_root = Path(__file__).parent.parent.parent.parent
            prompt_dirs = [str(project_root / "prompts")]

        self.prompt_engine = PromptEngine(
            template_dirs=prompt_dirs,
            auto_reload=True,
            cache_size=50,
        )

        self._llm_providers: dict[LLMProvider, Any] = {}
        self.default_provider = default_provider

        self.logger.info(
            "Initialised LLM Manager",
            extra={
                "prompt_dirs": prompt_dirs,
                "default_provider": default_provider.value,
            },
        )

    def register_provider(self, config: LLMProviderConfig) -> None:
        try:
            llm_instance = LLMFactory.create_llm(config)
            self._llm_providers[config.provider] = llm_instance

            self.logger.info(
                f"Registered {config.provider.value} LLM provider",
                extra={
                    "provider": config.provider.value,
                    "model": config.default_model,
                },
            )
        except Exception as e:
            self.logger.error(
                f"Failed to register {config.provider.value} provider",
                extra={
                    "provider": config.provider.value,
                    "error": str(e),
                },
            )
            raise

    async def generate_from_template(
        self,
        template_name: str,
        context: dict[str, Any],
        model: str | None = None,
        provider: LLMProvider | None = None,
        generation_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        provider = provider or self.default_provider

        if provider not in self._llm_providers:
            raise ValueError(f"Provider {provider.value} not registered")

        template_context = TemplateContext(variables=context)
        rendered_prompt = self.prompt_engine.render_template(
            template_name, template_context
        )

        messages = [LLMMessage(role=LLMRole.USER, content=rendered_prompt)]
        params = generation_params or {}
        llm_provider = self._llm_providers[provider]

        request = LLMRequest(
            messages=messages,
            model=model or llm_provider.config.default_model,
            **params,
        )

        self.logger.info(
            f"Generating response using template '{template_name}'",
            extra={
                "template_name": template_name,
                "provider": provider.value,
                "model": request.model,
                "context_keys": list(context.keys()),
            },
        )

        response = await llm_provider.generate(request)
        return cast(LLMResponse, response)

    async def cross_check_response(
        self,
        original_query: str,
        response_to_check: str,
        response_provider: str | None = None,
        validation_aspects: list[str] | None = None,
        checker_provider: LLMProvider | None = None,
    ) -> LLMResponse:
        context: dict[str, Any] = {
            "original_query": original_query,
            "llm_response": response_to_check,
        }

        if response_provider:
            context["response_provider"] = response_provider

        if validation_aspects:
            context["validation_aspects"] = ", ".join(validation_aspects)

        return await self.generate_from_template(
            template_name="crosscheck/response_validation",
            context=context,
            provider=checker_provider,
            generation_params={
                "temperature": 0.3,
                "max_tokens": 2000,
            },
        )

    async def code_review(
        self,
        code: str,
        language: str,
        focus_areas: list[str] | None = None,
        severity_threshold: str | None = None,
        include_suggestions: bool = True,
        provider: LLMProvider | None = None,
    ) -> LLMResponse:
        context: dict[str, Any] = {
            "code": code,
            "language": language,
            "include_suggestions": include_suggestions,
        }

        if focus_areas:
            context["focus_areas"] = focus_areas

        if severity_threshold:
            context["severity_threshold"] = severity_threshold

        return await self.generate_from_template(
            template_name="tasks/code_review",
            context=context,
            provider=provider,
            generation_params={
                "temperature": 0.2,
                "max_tokens": 3000,
            },
        )

    def list_available_templates(self, category: str | None = None) -> list[str]:
        return self.prompt_engine.list_templates(category=category)

    def get_available_providers(self) -> list[LLMProvider]:
        return list(self._llm_providers.keys())

    async def health_check(self) -> dict[str, Any]:
        health_status: dict[str, Any] = {
            "prompt_engine": {
                "status": "healthy",
                "template_dirs": [str(d) for d in self.prompt_engine.template_dirs],
                "templates_count": len(self.prompt_engine.list_templates()),
            },
            "providers": {},
        }

        providers = health_status["providers"]
        assert isinstance(providers, dict)

        for provider, llm_instance in self._llm_providers.items():
            try:
                provider_health = await llm_instance.health_check()
                providers[provider.value] = provider_health
            except Exception as e:
                providers[provider.value] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        return health_status
