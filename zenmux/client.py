"""ZenMux OpenAI-compatible client."""

from openai import AsyncOpenAI
from .models import ZenMuxEndpoint
from config import ZenMuxConfig


class ZenMuxOpenAIClient:
    """OpenAI-compatible client for ZenMux."""

    def __init__(self, config: ZenMuxConfig):
        self.config = config
        self._client = None

    def get_client(self, endpoint: ZenMuxEndpoint = None) -> AsyncOpenAI:
        """Get an AsyncOpenAI client configured for ZenMux with connection pooling."""
        if not self.config.api_key:
            raise ValueError(
                "ZENMUX_API_KEY environment variable is required. "
                "Please set it with your ZenMux API key."
            )

        # Reuse the same client instance for connection pooling
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
                timeout=600.0,
                max_retries=1,
                # Optimize connection pool for high concurrency
                http_client=None  # Use default with connection pooling
            )

        return self._client

    async def close(self):
        """Close the HTTP client and free resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Async context manager exit with cleanup."""
        await self.close()

    @staticmethod
    def format_model_name(model_slug: str, provider_slug: str) -> str:
        """Format model name for ZenMux API."""
        return f"{model_slug}:{provider_slug}"

    def supports_multimodal(self, endpoint: ZenMuxEndpoint) -> bool:
        """Check if endpoint supports multimodal inputs."""
        # This would depend on the model's capabilities
        # For now, we'll check the suitable_api field
        return "messages" in endpoint.suitable_api