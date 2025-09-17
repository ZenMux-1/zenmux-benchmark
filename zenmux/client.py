"""ZenMux OpenAI-compatible client."""

import os
from openai import AsyncOpenAI
from .models import ZenMuxEndpoint
from config import ZenMuxConfig


class ZenMuxOpenAIClient:
    """OpenAI-compatible client for ZenMux."""

    def __init__(self, config: ZenMuxConfig):
        self.config = config
        self._client = None

    def get_client(self, endpoint: ZenMuxEndpoint) -> AsyncOpenAI:
        """Get an AsyncOpenAI client configured for ZenMux."""
        if not self.config.api_key:
            raise ValueError(
                "ZENMUX_API_KEY environment variable is required. "
                "Please set it with your ZenMux API key."
            )

        return AsyncOpenAI(
            base_url=self.config.api_base_url,
            api_key=self.config.api_key,
            timeout=600.0,
            max_retries=1
        )

    @staticmethod
    def format_model_name(model_slug: str, provider_slug: str) -> str:
        """Format model name for ZenMux API."""
        return f"{model_slug}:{provider_slug}"

    def supports_multimodal(self, endpoint: ZenMuxEndpoint) -> bool:
        """Check if endpoint supports multimodal inputs."""
        # This would depend on the model's capabilities
        # For now, we'll check the suitable_api field
        return "messages" in endpoint.suitable_api