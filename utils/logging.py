"""Professional logging system for ZenMux Benchmark."""

import os
import logging
import sys
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path


class BenchmarkLogger:
    """Centralized logging system for ZenMux Benchmark."""

    _loggers: Dict[str, logging.Logger] = {}
    _handlers_setup: bool = False
    _log_dir: Optional[str] = None
    _batch_timestamp: Optional[str] = None

    @classmethod
    def setup_logging(
        cls,
        log_dir: str = "logs",
        batch_timestamp: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ) -> None:
        """Setup logging configuration for the entire application.

        Args:
            log_dir: Base directory for log files
            batch_timestamp: Timestamp for this batch run (creates subdirectory)
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        if cls._handlers_setup:
            return

        # Generate timestamp if not provided
        if batch_timestamp is None:
            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cls._batch_timestamp = batch_timestamp
        cls._log_dir = os.path.join(log_dir, batch_timestamp)

        # Create logs directory
        Path(cls._log_dir).mkdir(parents=True, exist_ok=True)

        # Setup formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Setup file handlers
        main_log_file = os.path.join(cls._log_dir, "zenmux_benchmark.log")
        error_log_file = os.path.join(cls._log_dir, "errors.log")

        # Main log file (all messages)
        file_handler = logging.FileHandler(main_log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(detailed_formatter)

        # Error log file (only errors and above)
        error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)

        # Suppress noisy HTTP debug logs
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

        # Suppress noisy HuggingFace datasets debug logs
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        logging.getLogger("fsspec.local").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)

        cls._handlers_setup = True

        # Log the setup
        setup_logger = cls.get_logger("logging")
        setup_logger.info(f"🔧 Logging system initialized")
        setup_logger.info(f"📁 Log directory: {cls._log_dir}")
        setup_logger.info(f"🕐 Batch timestamp: {batch_timestamp}")
        setup_logger.debug(f"📊 Console level: {logging.getLevelName(console_level)}")
        setup_logger.debug(f"📝 File level: {logging.getLevelName(file_level)}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger for a specific module.

        Args:
            name: Logger name (typically module name)

        Returns:
            Configured logger instance
        """
        if not cls._handlers_setup:
            # Auto-setup with defaults if not done yet
            cls.setup_logging()

        if name not in cls._loggers:
            logger = logging.getLogger(f"zenmux.{name}")
            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def create_model_logger(cls, model_identifier: str) -> logging.Logger:
        """Create a dedicated logger for a specific model evaluation.

        Args:
            model_identifier: Model identifier (e.g., 'openai/gpt-4o:openai')

        Returns:
            Dedicated logger with model-specific file handler
        """
        if not cls._handlers_setup:
            cls.setup_logging()

        # Sanitize model identifier for filename
        safe_name = model_identifier.replace("/", "_").replace(":", "_")
        logger_name = f"model.{safe_name}"

        if logger_name not in cls._loggers:
            logger = logging.getLogger(f"zenmux.{logger_name}")

            # Add model-specific file handler
            model_log_file = os.path.join(cls._log_dir, f"model_{safe_name}.log")
            model_handler = logging.FileHandler(model_log_file, mode='a', encoding='utf-8')
            model_handler.setLevel(logging.DEBUG)

            # Model-specific formatter
            model_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            model_handler.setFormatter(model_formatter)

            logger.addHandler(model_handler)
            cls._loggers[logger_name] = logger

            logger.info(f"🚀 Model logger created for {model_identifier}")

        return cls._loggers[logger_name]

    @classmethod
    def get_log_directory(cls) -> Optional[str]:
        """Get the current log directory path."""
        return cls._log_dir

    @classmethod
    def get_batch_timestamp(cls) -> Optional[str]:
        """Get the current batch timestamp."""
        return cls._batch_timestamp


# Convenience functions for common loggers
def get_evaluation_logger() -> logging.Logger:
    """Get logger for evaluation operations."""
    return BenchmarkLogger.get_logger("evaluation")


def get_judge_logger() -> logging.Logger:
    """Get logger for judge operations."""
    return BenchmarkLogger.get_logger("judge")


def get_runner_logger() -> logging.Logger:
    """Get logger for runner operations."""
    return BenchmarkLogger.get_logger("runner")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations."""
    return BenchmarkLogger.get_logger("api")


def get_model_logger(model_identifier: str) -> logging.Logger:
    """Get dedicated logger for a specific model."""
    return BenchmarkLogger.create_model_logger(model_identifier)


# Performance logging utilities
class PerformanceTimer:
    """Context manager for timing operations with automatic logging."""

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.log(self.level, f"🔄 Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time

        if exc_type is None:
            self.logger.log(self.level, f"✅ Completed {self.operation} in {elapsed:.2f}s")
        else:
            self.logger.error(f"❌ Failed {self.operation} after {elapsed:.2f}s: {exc_val}")


def log_performance(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """Decorator for timing function calls with automatic logging."""
    def decorator(func):
        import functools

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with PerformanceTimer(logger, f"{operation} ({func.__name__})", level):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with PerformanceTimer(logger, f"{operation} ({func.__name__})", level):
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Import asyncio for decorator
import asyncio