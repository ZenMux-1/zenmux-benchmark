"""ZenMux API client."""

import json
import requests
from typing import List, Optional
from .models import ZenMuxModel
from config import ZenMuxConfig


class ZenMuxAPI:
    """Client for ZenMux API."""

    def __init__(self, config: Optional[ZenMuxConfig] = None):
        self.config = config or ZenMuxConfig()

    def get_available_models(self) -> List[ZenMuxModel]:
        """Fetch all available models from ZenMux API."""
        try:
            response = requests.get(self.config.model_list_endpoint, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data.get("success", False):
                raise ValueError(f"API returned success=False: {data}")

            models = []
            for model_data in data.get("data", []):
                try:
                    model = ZenMuxModel.from_dict(model_data)
                    models.append(model)
                except Exception as e:
                    print(f"Warning: Failed to parse model {model_data.get('slug', 'unknown')}: {e}")
                    continue

            return models

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch models from ZenMux API: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON response: {e}")

    def filter_models_by_modality(
        self,
        models: List[ZenMuxModel],
        text_only: bool = False
    ) -> List[ZenMuxModel]:
        """Filter models based on input modality requirements."""
        if text_only:
            return [model for model in models if model.supports_text_only]
        return models

    def get_all_model_endpoint_pairs(
        self,
        text_only: bool = False
    ) -> List[tuple[str, ZenMuxModel, "ZenMuxEndpoint"]]:
        """Get all model:provider combinations."""
        models = self.get_available_models()
        filtered_models = self.filter_models_by_modality(models, text_only)

        pairs = []
        for model in filtered_models:
            # Only include models with visible=1
            if model.visible == 1:
                for model_id, endpoint in model.get_model_endpoint_pairs():
                    # Only include endpoints with visible=1
                    if endpoint.visible == 1:
                        pairs.append((model_id, model, endpoint))

        return pairs