"""Configuration management for ZenMux Benchmark."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ZenMuxConfig:
    """ZenMux API configuration."""

    api_base_url: str = "https://zenmux.ai/api/v1"
    api_key: Optional[str] = None
    model_list_endpoint: str = "https://zenmux.ai/api/frontend/model/available/list"

    def __post_init__(self):
        # Load from environment variables if not provided
        if self.api_key is None:
            self.api_key = os.getenv("ZENMUX_API_KEY")

        # Override with environment variable if set
        env_base_url = os.getenv("ZENMUX_API_BASE_URL")
        if env_base_url:
            self.api_base_url = env_base_url


@dataclass
class HLEConfig:
    """HLE evaluation configuration."""

    dataset_name: str = "cais/hle"
    dataset_split: str = "test"
    judge_model: str = "openai/gpt-5:openai"
    max_completion_tokens: int = 8192
    temperature: float = 0.0
    num_workers: int = 10
    timeout: float = 600.0
    max_retries: int = 1


@dataclass
class BenchmarkConfig:
    """Overall benchmark configuration."""

    zenmux: ZenMuxConfig
    hle: HLEConfig

    # Output directories
    output_dir: str = "results"
    predictions_dir: str = "predictions"
    judged_dir: str = "judged"

    def __post_init__(self):
        # Don't create directories automatically to avoid unwanted directory creation
        pass

    def create_directories(self):
        """Create the configured directories."""
        for dir_path in [self.output_dir, self.predictions_dir, self.judged_dir]:
            os.makedirs(dir_path, exist_ok=True)


def get_config() -> BenchmarkConfig:
    """Get the default benchmark configuration."""
    return BenchmarkConfig(
        zenmux=ZenMuxConfig(),
        hle=HLEConfig()
    )