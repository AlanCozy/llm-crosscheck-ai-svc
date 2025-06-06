"""
CrossCheck service for LLM response validation and analysis.

This service provides high-level methods for cross-checking LLM responses,
demonstrating practical usage of the LLM abstraction layer and prompt templates.
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..core.logging import LoggerMixin
from ..schemas.llm import LLMProvider, LLMProviderConfig, LLMResponse
from .llm_manager import LLMManager


class CrossCheckService(LoggerMixin):
    """
    Service for cross-checking and validating LLM responses.

    Provides high-level methods for response validation, multi-provider
    comparison, and comprehensive analysis workflows.
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """
        Initialise the CrossCheck service.

        Args:
            llm_manager: Optional LLM manager instance
        """
        self.llm_manager = llm_manager or LLMManager()

        self.logger.info("Initialised CrossCheck service")

    def configure_providers(self, provider_configs: List[Dict[str, Any]]) -> None:
        """
        Configure multiple LLM providers from configuration.

        Args:
            provider_configs: List of provider configuration dictionaries
        """
        for config_dict in provider_configs:
            try:
                # Convert dict to LLMProviderConfig
                config = LLMProviderConfig(**config_dict)
                self.llm_manager.register_provider(config)

                self.logger.info(
                    f"Configured {config.provider.value} provider",
                    extra={"provider": config.provider.value},
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to configure provider",
                    extra={"config": config_dict, "error": str(e)},
                )

    async def validate_response(
        self,
        query: str,
        response: str,
        response_provider: Optional[str] = None,
        validation_criteria: Optional[List[str]] = None,
        validator_provider: Optional[LLMProvider] = None,
    ) -> Dict[str, Any]:
        """
        Validate an LLM response using cross-checking.

        Args:
            query: Original query
            response: Response to validate
            response_provider: Provider that generated the response
            validation_criteria: Specific validation criteria
            validator_provider: Provider to use for validation

        Returns:
            Validation results with analysis
        """
        self.logger.info(
            "Starting response validation",
            extra={
                "query_length": len(query),
                "response_length": len(response),
                "response_provider": response_provider,
            },
        )

        try:
            # Perform cross-check validation
            validation_response = await self.llm_manager.cross_check_response(
                original_query=query,
                response_to_check=response,
                response_provider=response_provider,
                validation_aspects=validation_criteria,
                checker_provider=validator_provider,
            )

            # Parse validation results (simplified parsing)
            validation_content = validation_response.content

            # Extract key metrics (this would be more sophisticated in production)
            confidence_score = self._extract_confidence_score(validation_content)
            overall_assessment = self._extract_overall_assessment(validation_content)

            results = {
                "validation_id": str(validation_response.request_id),
                "overall_assessment": overall_assessment,
                "confidence_score": confidence_score,
                "detailed_analysis": validation_content,
                "validator_provider": validation_response.provider.value,
                "validation_model": validation_response.model,
                "response_time_ms": validation_response.response_time_ms,
                "token_usage": {
                    "prompt_tokens": validation_response.usage.prompt_tokens
                    if validation_response.usage
                    else None,
                    "completion_tokens": validation_response.usage.completion_tokens
                    if validation_response.usage
                    else None,
                    "total_tokens": validation_response.usage.total_tokens
                    if validation_response.usage
                    else None,
                },
            }

            self.logger.info(
                "Completed response validation",
                extra={
                    "validation_id": results["validation_id"],
                    "overall_assessment": overall_assessment,
                    "confidence_score": confidence_score,
                },
            )

            return results

        except Exception as e:
            self.logger.error("Failed to validate response", extra={"error": str(e)})
            raise

    def _extract_confidence_score(self, validation_content: str) -> Optional[float]:
        """Extract confidence score from validation content."""
        # Simple regex-based extraction (would be more sophisticated in production)
        import re

        patterns = [
            r"confidence.*?(\d+(?:\.\d+)?)",
            r"score.*?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:/\s*10|out\s+of\s+10)",
        ]

        for pattern in patterns:
            match = re.search(pattern, validation_content.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalise to 0-1 scale if needed
                    if score > 1:
                        score = score / 10
                    return min(max(score, 0), 1)  # Clamp to 0-1
                except ValueError:
                    continue

        return None

    def _extract_overall_assessment(self, validation_content: str) -> str:
        """Extract overall assessment from validation content."""
        # Simple keyword-based extraction
        content_lower = validation_content.lower()

        if "excellent" in content_lower:
            return "Excellent"
        elif "good" in content_lower:
            return "Good"
        elif "fair" in content_lower:
            return "Fair"
        elif "poor" in content_lower:
            return "Poor"
        else:
            return "Unknown"
