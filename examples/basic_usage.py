#!/usr/bin/env python3
"""
Basic usage example for LLM CrossCheck AI Service.

This script demonstrates how to use the LLM abstraction layer and prompt engine
for cross-checking and validating LLM responses.
"""

import asyncio
import os

# Add src to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_crosscheck.schemas.llm import LLMProvider, LLMProviderConfig
from llm_crosscheck.services.crosscheck_service import CrossCheckService
from llm_crosscheck.services.llm_manager import LLMManager


async def main():
    """Demonstrate basic usage of the LLM CrossCheck service."""

    print("🚀 LLM CrossCheck AI Service - Basic Usage Example")
    print("=" * 60)

    # Initialize LLM Manager
    print("\n1. Initialising LLM Manager...")
    llm_manager = LLMManager()

    # Configure providers (you'll need to set these environment variables)
    providers_configured = 0

    # Configure OpenAI if API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("   ✓ Configuring OpenAI provider...")
        openai_config = LLMProviderConfig(
            provider=LLMProvider.OPENAI,
            api_key=openai_key,
            default_model="gpt-3.5-turbo",  # Using cheaper model for demo
            available_models=["gpt-3.5-turbo", "gpt-4"],
        )
        llm_manager.register_provider(openai_config)
        providers_configured += 1
    else:
        print("   ⚠️  OpenAI API key not found (set OPENAI_API_KEY)")

    # Configure Anthropic if API key is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("   ✓ Configuring Anthropic provider...")
        anthropic_config = LLMProviderConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=anthropic_key,
            default_model="claude-3-haiku-20240307",  # Using cheaper model for demo
            available_models=["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
        )
        llm_manager.register_provider(anthropic_config)
        providers_configured += 1
    else:
        print("   ⚠️  Anthropic API key not found (set ANTHROPIC_API_KEY)")

    if providers_configured == 0:
        print("\n❌ No LLM providers configured. Please set API keys:")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
        return

    print(f"\n✅ Configured {providers_configured} LLM provider(s)")

    # Example 1: Basic template usage
    print("\n2. Testing Basic Template Usage...")
    try:
        response = await llm_manager.generate_from_template(
            template_name="system/assistant",
            context={
                "assistant_name": "CrossCheck AI Demo",
                "capabilities": [
                    "Response validation",
                    "Code review",
                    "Multi-provider comparison",
                    "Template-based prompt management",
                ],
                "tone": "friendly and professional",
            },
        )

        print("   ✓ Generated system prompt response:")
        print(f"   Provider: {response.provider.value}")
        print(f"   Model: {response.model}")
        print(f"   Response length: {len(response.content)} characters")
        print(f"   Response time: {response.response_time_ms:.1f}ms")
        if response.usage:
            print(f"   Tokens used: {response.usage.total_tokens}")
        print(f"\n   Content preview: {response.content[:200]}...")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("🎉 Demo completed! Check the output above for results.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
