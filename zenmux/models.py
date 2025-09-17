"""ZenMux data models."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ZenMuxEndpoint:
    """Represents a ZenMux model endpoint."""

    pricing_completion: str
    pricing_prompt: str
    context_length: int
    provider: str
    provider_slug: str
    max_completion_tokens: Optional[int]
    supports_streaming: bool
    supports_reasoning: bool
    supports_tool_parameters: bool
    supported_parameters: List[str]
    can_abort: bool
    visible: int
    suitable_api: str


@dataclass
class ZenMuxModel:
    """Represents a ZenMux model."""

    id: str
    name: str
    description: str
    author: str
    slug: str
    input_modalities: List[str]
    publish_time: str
    icon_url: str
    visible: int
    endpoints: List[ZenMuxEndpoint]

    @property
    def supports_images(self) -> bool:
        """Check if the model supports image inputs."""
        return "image" in self.input_modalities

    @property
    def supports_text_only(self) -> bool:
        """Check if the model supports text-only inputs."""
        return "text" in self.input_modalities

    def get_model_endpoint_pairs(self) -> List[tuple[str, ZenMuxEndpoint]]:
        """Get all model:provider combinations for this model."""
        pairs = []
        for endpoint in self.endpoints:
            model_identifier = f"{self.slug}:{endpoint.provider_slug}"
            pairs.append((model_identifier, endpoint))
        return pairs

    @classmethod
    def from_dict(cls, data: dict) -> "ZenMuxModel":
        """Create ZenMuxModel from dictionary."""
        endpoints = [
            ZenMuxEndpoint(**endpoint_data)
            for endpoint_data in data["endpoints"]
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            author=data["author"],
            slug=data["slug"],
            input_modalities=data["input_modalities"],
            publish_time=data["publish_time"],
            icon_url=data["icon_url"],
            visible=data["visible"],
            endpoints=endpoints
        )