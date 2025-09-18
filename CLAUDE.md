# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `zenmux-benchmark`, an open-source AI model performance evaluation framework by ZenMux. It provides a production-grade evaluation framework for comprehensive testing of AI models on the Humanity's Last Exam (HLE) dataset through the ZenMux unified API platform.

## Development Environment

- **Python Version**: 3.13+ (specified in pyproject.toml)
- **Dependency Management**: Uses `uv` instead of traditional pip/venv
- **Virtual Environment**: Located in `.venv/` (managed by uv)

## Common Commands

### Running Evaluations

```bash
# Run basic test
uv run python example.py

# Small-scale test (recommended first run)
uv run python benchmark.py --text-only --max-samples 5

# Full evaluation of all models
uv run python benchmark.py

# Evaluate specific model
uv run python benchmark.py --mode single --model-slug openai/gpt-4o --provider-slug openai

# Filter models and exclude certain ones
uv run python benchmark.py --model-filter gpt --exclude anthropic openai/gpt-4o-mini

# Test runs with different concurrency settings
uv run python benchmark.py --num-workers 5 --max-samples 10

# Fix evaluation or judge failures from previous runs
uv run python benchmark.py --fix-evaluation results/20250919_011623
uv run python benchmark.py --fix-judge results/20250919_011623

# Calculate metrics only for complete models
uv run python benchmark.py --metrics-only results/20250919_011623
```

### Running Tests

```bash
# Run all tests from tests directory
uv run python -m pytest tests/

# Run specific test file
uv run python tests/test_exclude_functionality.py

# Run exclude functionality tests specifically
uv run python tests/test_exclude_functionality.py
```

### Development Setup

```bash
# Install dependencies
uv sync

# Add new dependencies
uv add <package-name>
```

## Architecture Overview

The project implements a **dual-layer concurrent architecture** with timestamped result organization:

### Core Components

1. **ZenMux API Layer** (`zenmux/`)
   - `api.py`: ZenMux API integration and model discovery
   - `client.py`: HTTP client with connection pooling for ZenMux services
   - `models.py`: Data models for API responses and configurations

2. **HLE Evaluation Engine** (`hle/`)
   - `runner.py`: **Main orchestrator** with dual-layer concurrency (model-level + request-level)
   - `evaluation.py`: Individual model evaluation logic with resumable execution
   - `judge.py`: Automated scoring using judge models with structured response parsing
   - `dataset.py`: HLE dataset loading and question formatting

3. **Configuration Management** (`config.py`)
   - Centralized configuration for concurrency, API settings, and output paths
   - Support for timestamped directory structure
   - Environment variable integration

### Dual-Layer Concurrency Architecture

The framework implements sophisticated concurrency control:

- **Outer Layer** (`max_concurrent_models`): Multiple models evaluated simultaneously
- **Inner Layer** (`num_workers`): Concurrent requests per model
- **Example**: `max_concurrent_models=5, num_workers=10` = 50 total concurrent API calls

### File Organization System

Results are automatically organized with timestamps:
```
results/
├── 20250917_173456/          # Timestamped run directory
│   ├── predictions/          # Model prediction files
│   ├── judged/              # Scored results
│   ├── question_ids_*.json  # Question IDs used in run
│   ├── available_models_*.json # Available models list
│   ├── evaluation_failures_*.json # Evaluation failure tracking
│   ├── judge_failures_*.json # Judge failure tracking
│   ├── metrics_summary_*.json # Aggregated metrics and results
│   └── metrics_only_*.json  # Independent metrics calculation
```

### Key Data Flow

1. **Model Discovery**: ZenMux API provides available models across providers
2. **Concurrent Evaluation**: Models evaluated with dual-layer concurrency
3. **Automatic Judging**: Responses scored using structured judge models
4. **Failure Tracking**: Comprehensive tracking of evaluation and judge failures
5. **Result Aggregation**: Metrics compiled with confidence intervals and calibration
6. **Quality Assurance**: Strict validation ensures metrics only include complete data

## Configuration

### Core Settings (config.py)

```python
class HLEConfig:
    num_workers: int = 10              # Inner concurrency: requests per model
    max_concurrent_models: int = 5     # Outer concurrency: simultaneous models
    judge_model: str = "openai/gpt-5:openai"  # Model used for scoring
    timeout: float = 600.0             # Request timeout in seconds
```

### Environment Variables

```bash
export ZENMUX_API_KEY="your_api_key"  # Required for all operations
```

## Important Implementation Details

### Connection Management
- HTTP client reuses connections via connection pooling
- Automatic retry mechanisms with exponential backoff
- Resource cleanup and proper connection closing

### Resumable Execution
- Evaluations can be resumed if interrupted
- Existing predictions and judgments are preserved
- File-based state management with JSON metadata

### Model Exclusion System
- Supports exact model matching (`openai/gpt-4o`)
- Vendor-level exclusion (`anthropic` excludes all Anthropic models)
- Model-name only matching (`gpt-4o` matches across providers)

### Error Handling
- Graceful degradation when models fail
- Detailed error reporting in results
- Continuation of evaluation despite individual failures
- Comprehensive failure tracking with separate fix operations
- Strict metrics validation excluding incomplete data

## Test Organization

All test scripts are located in `tests/` directory:
- `test_exclude_functionality.py`: Tests for model exclusion logic
- Test files should be placed in `tests/` directory, not in the root
