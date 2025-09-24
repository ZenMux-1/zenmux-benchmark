"""Configuration management for ZenMux Benchmark."""

import os
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class ZenMuxConfig:
    """ZenMux API configuration."""

    api_base_url: str = "https://zenmux.ai/api/v1"
    api_key: Optional[str] = None
    model_list_endpoint: str = "https://zenmux.ai/api/frontend/model/available/list"
    timeout: float = 300.0  # API request timeout - increased to handle slow responses
    max_retries: int = 2  # Increased retries for network issues
    enable_streaming: bool = True  # Enable streaming responses (default: True for faster TTFB)

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
    max_completion_tokens: Optional[int] = None
    judge_max_completion_tokens: Optional[int] = None  # Max completion tokens for judge model
    temperature: float = 0.0
    num_workers: int = 100  # Inner concurrency: requests per model (restored to high concurrency)
    max_concurrent_models: int = 10  # Outer concurrency: simultaneous models (restored to high concurrency)
    print_streaming_output: bool = True  # If True, print streaming responses to console in real-time


@dataclass
class BenchmarkConfig:
    """Overall benchmark configuration."""

    zenmux: ZenMuxConfig
    hle: HLEConfig

    # Output directories
    output_dir: str = "results"
    predictions_dir: str = "predictions"  # Will be relative to timestamped folder
    judged_dir: str = "judged"           # Will be relative to timestamped folder
    logs_dir: str = "logs"               # Logs directory

    # Logging configuration
    console_log_level: int = logging.INFO
    file_log_level: int = logging.INFO
    enable_model_specific_logs: bool = False  # If True, creates separate log file for each model

    # Timestamped run directory (set during initialization)
    run_dir: str = None
    batch_timestamp: str = None

    def __post_init__(self):
        # Don't create directories automatically to avoid unwanted directory creation
        pass

    def setup_timestamped_directories(self, timestamp: str = None) -> str:
        """Setup timestamped directory structure and initialize logging system."""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.batch_timestamp = timestamp
        self.run_dir = os.path.join(self.output_dir, timestamp)

        # Update paths to be within the timestamped directory
        timestamped_predictions_dir = os.path.join(self.run_dir, self.predictions_dir)
        timestamped_judged_dir = os.path.join(self.run_dir, self.judged_dir)

        # Create the directory structure
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(timestamped_predictions_dir, exist_ok=True)
        os.makedirs(timestamped_judged_dir, exist_ok=True)

        # Initialize logging system
        from utils.logging import BenchmarkLogger
        BenchmarkLogger.setup_logging(
            log_dir=self.logs_dir,
            batch_timestamp=timestamp,
            console_level=self.console_log_level,
            file_level=self.file_log_level,
            enable_model_specific_logs=self.enable_model_specific_logs
        )

        # Use logger instead of print
        from utils.logging import get_runner_logger
        logger = get_runner_logger()
        logger.info(f"📁 Created timestamped directories:")
        logger.info(f"   Run directory: {self.run_dir}")
        logger.info(f"   Predictions: {timestamped_predictions_dir}")
        logger.info(f"   Judged: {timestamped_judged_dir}")

        return self.run_dir

    def get_predictions_dir(self) -> str:
        """Get the full path to predictions directory."""
        if self.run_dir:
            return os.path.join(self.run_dir, self.predictions_dir)
        return self.predictions_dir

    def get_judged_dir(self) -> str:
        """Get the full path to judged directory."""
        if self.run_dir:
            return os.path.join(self.run_dir, self.judged_dir)
        return self.judged_dir

    def create_directories(self):
        """Create the configured directories (legacy method)."""
        for dir_path in [self.output_dir, self.predictions_dir, self.judged_dir]:
            os.makedirs(dir_path, exist_ok=True)


def get_config() -> BenchmarkConfig:
    """Get the default benchmark configuration."""
    return BenchmarkConfig(
        zenmux=ZenMuxConfig(),
        hle=HLEConfig()
    )