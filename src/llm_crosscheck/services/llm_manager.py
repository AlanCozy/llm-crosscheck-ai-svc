"""
LLM Manager service for orchestrating multiple LLM providers and prompt templates.

This service demonstrates how to use the LLM abstraction layer and prompt engine
together to create a comprehensive LLM management system.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """
    Comprehensive LLM management service.

    Orchestrates multiple LLM providers and manages prompt templates
    for consistent and reliable LLM interactions.
    """

    def __init__(
        self,
        prompt_dirs: Optional[List[str]] = None,
        default_provider: LLMProvider = LLMProvider.OPENAI,
    ):
        """
        Initialise the LLM Manager.

        Args:
            prompt_dirs: List of directories containing prompt templates
            default_provider: Default LLM provider to use
        """
        # Set up prompt directories
        if prompt_dirs is None:
            # Default to prompts directory at project root
            project_root = Path(__file__).parent.parent.parent.parent
            prompt_dirs = [str(project_root / "prompts")]

        # Initialise prompt engine
        self.prompt_engine = PromptEngine(
            template_dirs=prompt_dirs,
            auto_reload=True,  # Enable auto-reload for development
            cache_size=50,
        )

        # LLM provider instances
        self._llm_providers: Dict[LLMProvider, Any] = {}
        self.default_provider = default_provider

        self.logger.info(
            "Initialised LLM Manager",
            extra={
                "prompt_dirs": prompt_dirs,
                "default_provider": default_provider.value,
            },
        )

    def register_provider(self, config: LLMProviderConfig) -> None:
        """
        Register an LLM provider.

        Args:
            config: Provider configuration
        """
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
        context: Dict[str, Any],
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate LLM response using a prompt template.

        Args:
            template_name: Name of the prompt template
            context: Template context variables
            model: Specific model to use
            provider: Specific provider to use
            generation_params: Additional generation parameters

        Returns:
            LLM response
        """
        # Determine provider
        provider = provider or self.default_provider

        if provider not in self._llm_providers:
            raise ValueError(f"Provider {provider.value} not registered")

        # Render template
        template_context = TemplateContext(variables=context)
        rendered_prompt = self.prompt_engine.render_template(
            template_name, template_context
        )

        # Create LLM request
        messages = [LLMMessage(role=LLMRole.USER, content=rendered_prompt)]

        # Set up generation parameters
        params = generation_params or {}
        llm_provider = self._llm_providers[provider]

        request = LLMRequest(
            messages=messages,
            model=model or llm_provider.config.default_model,
            **params,
        )

        # Generate response
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

        self.logger.info(
            f"Generated response from template '{template_name}'",
            extra={
                "template_name": template_name,
                "provider": provider.value,
                "response_id": response.id,
                "response_length": len(response.content),
            },
        )

        return response

    async def cross_check_response(
        self,
        original_query: str,
        response_to_check: str,
        response_provider: Optional[str] = None,
        validation_aspects: Optional[List[str]] = None,
        checker_provider: Optional[LLMProvider] = None,
    ) -> LLMResponse:
        """
        Cross-check an LLM response using the validation template.

        Args:
            original_query: The original query that was asked
            response_to_check: The LLM response to validate
            response_provider: Name of the provider that generated the response
            validation_aspects: Additional aspects to validate
            checker_provider: Provider to use for cross-checking

        Returns:
            Cross-check analysis response
        """
        context = {
            "original_query": original_query,
            "llm_response": response_to_check,
        }

        if response_provider:
            context["response_provider"] = response_provider

        if validation_aspects:
            context["validation_aspects"] = validation_aspects

        return await self.generate_from_template(
            template_name="crosscheck/response_validation",
            context=context,
            provider=checker_provider,
            generation_params={
                "temperature": 0.3,  # Lower temperature for more consistent analysis
                "max_tokens": 2000,
            },
        )

    async def code_review(
        self,
        code: str,
        language: str,
        focus_areas: Optional[List[str]] = None,
        severity_threshold: Optional[str] = None,
        include_suggestions: bool = True,
        provider: Optional[LLMProvider] = None,
    ) -> LLMResponse:
        """
        Perform code review using the code review template.

        Args:
            code: Code to review
            language: Programming language
            focus_areas: Specific areas to focus on
            severity_threshold: Minimum severity level to report
            include_suggestions: Whether to include improvement suggestions
            provider: Provider to use for review

        Returns:
            Code review analysis response
        """
        context = {
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
                "temperature": 0.2,  # Lower temperature for consistent analysis
                "max_tokens": 3000,
            },
        )

    def list_available_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List available prompt templates.

        Args:
            category: Optional category filter

        Returns:
            List of template names
        """
        return self.prompt_engine.list_templates(category=category)

    def get_available_providers(self) -> List[LLMProvider]:
        """
        Get list of registered providers.

        Returns:
            List of registered provider identifiers
        """
        return list(self._llm_providers.keys())

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all registered providers.

        Returns:
            Health status for all providers
        """
        health_status = {
            "prompt_engine": {
                "status": "healthy",
                "template_dirs": [str(d) for d in self.prompt_engine.template_dirs],
                "templates_count": len(self.prompt_engine.list_templates()),
            },
            "providers": {},
        }

        for provider, llm_instance in self._llm_providers.items():
            try:
                provider_health = await llm_instance.health_check()
                health_status["providers"][provider.value] = provider_health
            except Exception as e:
                health_status["providers"][provider.value] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        return health_status
