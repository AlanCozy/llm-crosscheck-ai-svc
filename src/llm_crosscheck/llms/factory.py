"""
Factory for creating LLM provider instances.

This module provides a centralised factory for creating and configuring
different LLM provider instances based on configuration.
"""

from typing import Dict, Type

from ..models.llm import LLMProvider, LLMProviderConfig
from .base import BaseLLM, LLMError, LLMValidationError
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM


class LLMFactory:
    """Factory for creating LLM provider instances."""
    
    # Registry of available LLM providers
    _providers: Dict[LLMProvider, Type[BaseLLM]] = {
        LLMProvider.OPENAI: OpenAILLM,
        LLMProvider.ANTHROPIC: AnthropicLLM,
    }
    
    @classmethod
    def create_llm(cls, config: LLMProviderConfig) -> BaseLLM:
        """
        Create an LLM provider instance based on configuration.
        
        Args:
            config: Provider configuration
            
        Returns:
            Configured LLM provider instance
            
        Raises:
            LLMValidationError: If provider is not supported
            LLMError: If configuration is invalid
        """
        provider_class = cls._providers.get(config.provider)
        
        if not provider_class:
            available_providers = list(cls._providers.keys())
            raise LLMValidationError(
                f"Unsupported provider {config.provider}. "
                f"Available providers: {[p.value for p in available_providers]}"
            )
        
        try:
            return provider_class(config)
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            else:
                raise LLMError(f"Failed to create {config.provider} LLM: {str(e)}")
    
    @classmethod
    def register_provider(cls, provider: LLMProvider, provider_class: Type[BaseLLM]) -> None:
        """
        Register a new LLM provider.
        
        Args:
            provider: Provider identifier
            provider_class: Provider implementation class
        """
        if not issubclass(provider_class, BaseLLM):
            raise LLMValidationError(f"Provider class must inherit from BaseLLM")
        
        cls._providers[provider] = provider_class
    
    @classmethod
    def get_available_providers(cls) -> list[LLMProvider]:
        """
        Get list of available providers.
        
        Returns:
            List of available provider identifiers
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_provider_available(cls, provider: LLMProvider) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider: Provider identifier
            
        Returns:
            True if provider is available, False otherwise
        """
        return provider in cls._providers


# Additional provider implementations can be registered here
# when they're available and properly implemented

def _register_optional_providers() -> None:
    """Register optional providers that may not always be available."""
    try:
        # Try to import and register Mistral if available
        from .mistral_llm import MistralLLM
        LLMFactory.register_provider(LLMProvider.MISTRAL, MistralLLM)
    except ImportError:
        pass  # Mistral provider not available
    
    try:
        # Try to import and register Llama if available
        from .llama_llm import LlamaLLM
        LLMFactory.register_provider(LLMProvider.LLAMA, LlamaLLM)
    except ImportError:
        pass  # Llama provider not available
    
    try:
        # Try to import and register Azure OpenAI if available
        from .azure_llm import AzureOpenAILLM
        LLMFactory.register_provider(LLMProvider.AZURE_OPENAI, AzureOpenAILLM)
    except ImportError:
        pass  # Azure OpenAI provider not available
    
    try:
        # Try to import and register Hugging Face if available
        from .huggingface_llm import HuggingFaceLLM
        LLMFactory.register_provider(LLMProvider.HUGGING_FACE, HuggingFaceLLM)
    except ImportError:
        pass  # Hugging Face provider not available


# Register optional providers
_register_optional_providers() 