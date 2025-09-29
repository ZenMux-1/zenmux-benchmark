# ZenMux Benchmark

The open-source AI model performance evaluation framework by ZenMux. Official benchmark suite for evaluating AI model performance across multiple dimensions using the Humanity's Last Exam (HLE) dataset.

## Overview

ZenMux Benchmark is a production-grade evaluation framework that enables comprehensive testing of AI models on the Humanity's Last Exam (HLE) dataset through the ZenMux unified API platform. It provides seamless integration with ZenMux's API to evaluate models from multiple providers including OpenAI, Anthropic, Google, DeepSeek, and more.

### Key Features

- 🌐 **Unified API Access**: Evaluate models from multiple providers through ZenMux's single API
- 🧠 **HLE Integration**: Built-in support for Humanity's Last Exam, a comprehensive AI benchmark dataset
- 🔄 **Automatic Judging**: Intelligent scoring system using advanced judge models with structured output parsing
- 📊 **Text-Only Mode**: Filter multimodal questions for text-only model evaluation
- 🚫 **Smart Model Exclusion**: Dual exclusion system with `--exclude-model` (vendor/model filtering) and `--exclude-provider` (provider-based filtering)
- ⚡ **Dual-Layer Concurrency**: Advanced parallel processing with model-level and request-level concurrency
- 🔧 **Failure Recovery**: Intelligent fix system for recovering from evaluation and judge failures
- ⚙️ **Production Ready**: Robust error handling, retry mechanisms, and resumable evaluations
- 📈 **Comprehensive Statistics**: Detailed statistics with failure tracking and performance metrics
- 🚀 **CI/CD Support**: GitHub Actions integration for automated benchmarking

## Quick Start

### 1. Environment Setup

Set up your ZenMux API key:

```bash
# Set ZenMux API key (required)
export ZENMUX_API_KEY="your_zenmux_api_key"

# Note: Judging also uses ZenMux API, no separate OpenAI API key needed
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Basic Testing (Recommended First Run)

```bash
# Test basic functionality
uv run python example.py

# Small-scale test - text-only questions, 3 samples
uv run python benchmark.py --text-only --max-samples 3

# Test specific model
uv run python benchmark.py --mode single \
  --model-slug openai/gpt-4o-mini \
  --provider-slug openai \
  --text-only --max-samples 10

# Test without judging (prediction only)
uv run python benchmark.py --mode single \
  --model-slug openai/gpt-4o-mini \
  --provider-slug openai \
  --text-only --max-samples 10 \
  --no-judge

# Test all models except expensive ones
uv run python benchmark.py --text-only --max-samples 5 --exclude-model openai/gpt-4o

# Use moderate calculation to include models with partial judgment failures
uv run python benchmark.py --text-only --max-samples 10 --moderate-calculate

# Resume interrupted evaluation
uv run python benchmark.py --mode all --resume results/20250922_122904
```

## Core Functionality

### 1. Evaluate All Models

```bash
# Evaluate all available models (takes a long time!)
uv run python benchmark.py --mode all

# Text-only questions only
uv run python benchmark.py --mode all --text-only

# Limit samples for quick testing
uv run python benchmark.py --mode all --text-only --max-samples 10
```

### 2. Evaluate Specific Models

```bash
# Evaluate specific model with specific provider
uv run python benchmark.py --mode single \
  --model-slug deepseek/deepseek-chat \
  --provider-slug deepseek

# The system automatically evaluates all available endpoints for the model
```

### 3. Filtered Evaluation

```bash
# Evaluate all DeepSeek models
uv run python benchmark.py --mode filter --model-filter deepseek --text-only

# Evaluate all Claude models, text-only questions
uv run python benchmark.py --mode filter --model-filter claude --text-only

# Evaluate all models from OpenAI provider
uv run python benchmark.py --mode filter --model-filter openai
```

### 4. Model Exclusion

```bash
# Exclude specific models (all providers for this model)
uv run python benchmark.py --exclude-model openai/gpt-4o anthropic/claude-3-haiku

# Exclude entire vendors (all models from vendor)
uv run python benchmark.py --exclude-model anthropic openai

# Exclude specific provider (all models using this provider)
uv run python benchmark.py --exclude-provider theta

# Exclude specific model from specific provider only
uv run python benchmark.py --exclude-model openai/gpt-4o:openai

# Combine both exclusion types
uv run python benchmark.py --exclude-provider theta \
  --exclude-model google/gemini-2.5-flash-lite:google-vertex anthropic/claude-opus-4.1 \
  --text-only --max-samples 3

# Combine with other options
uv run python benchmark.py --mode filter --model-filter gpt --exclude-model openai/gpt-4o-mini

# Use moderate calculation for research and analysis
uv run python benchmark.py --mode all --moderate-calculate --text-only
```

### 5. Moderate Calculation Mode

For research purposes or when you want to maximize data utilization from partially successful runs:

```bash
# Enable moderate calculation - includes models with partial judgment failures
uv run python benchmark.py --moderate-calculate --text-only

# Compare strict vs moderate results
uv run python benchmark.py --text-only  # strict mode (default)
uv run python benchmark.py --text-only --moderate-calculate  # moderate mode

# Fix failed judgments, then use moderate calculation
uv run python benchmark.py --fix results/20250922_122904
uv run python benchmark.py --moderate-calculate --resume results/20250922_122904
```

### 6. Resume Interrupted Evaluations

If your evaluation was interrupted (e.g., network issues, system restart), you can resume from where it left off:

```bash
# Resume evaluation from a specific timestamp directory
uv run python benchmark.py --mode all --exclude-provider theta --text-only --resume results/20250922_122904

# Resume with specific parameters (same as original run)
uv run python benchmark.py --mode all --text-only --resume results/20250922_122904

# Resume filtered evaluation
uv run python benchmark.py --mode filter --model-filter gpt --resume results/20250922_122904

# Resume single model evaluation
uv run python benchmark.py --mode single \
  --model-slug google/gemini-2.5-flash --provider-slug skyrouter \
  --text-only \
  --resume results/20250922_122904

# The system will:
# - Use the existing timestamped directory
# - Skip models that are already completed
# - Continue evaluating incomplete models
# - Skip questions already answered within each model
# - Continue with judging and metrics calculation
```

### 7. Failure Recovery (Advanced)

When evaluations encounter failures, you can automatically retry and fix them using concurrent processing:

```bash
# Fix both evaluation and judge failures from a previous run
uv run python benchmark.py --fix results/20250922_093840

# The system will:
# - Read models from available_models_*.json
# - Process multiple models concurrently (up to max_concurrent_models)
# - For each model, fix failed questions concurrently (up to num_workers)
# - Retry evaluation failures first, then judge failures
# - Update files with successful results
# - Recalculate metrics for all models

# Adjust concurrency for fix operations
uv run python benchmark.py --fix results/20250922_093840 --num-workers 5

# The fix mode uses the same dual-layer concurrency as normal evaluation:
# - Outer layer: Multiple models fixed simultaneously
# - Inner layer: Multiple questions per model fixed concurrently
```

## Important Options

### `--text-only`

- Filters out questions containing images, evaluates text-only questions
- Suitable for models that don't support multimodal inputs
- Significantly reduces evaluation time and cost

### `--max-samples N`

- Limits evaluation to the first N questions
- Perfect for testing and debugging
- Recommended to start with small values (3-10)

### `--no-judge`

- Skips automatic judging
- Use when you want to manually score results or just generate predictions

### `--num-workers N`

- Controls concurrent requests per model (inner concurrency)
- Default is 3, adjust based on API rate limits and provider capabilities
- Each model processes questions concurrently with this limit

### `--exclude-model MODEL1 MODEL2 ...`

- Exclude specific models from evaluation
- Supports three matching patterns:
  - **Exact match**: `openai/gpt-4o` (excludes this model from all providers)
  - **Specific provider**: `openai/gpt-4o:openai` (excludes only from OpenAI provider)
  - **Vendor exclusion**: `anthropic` (excludes all models from Anthropic vendor)
- Can combine multiple exclusion patterns

### `--exclude-provider PROVIDER1 PROVIDER2 ...`

- Exclude all models from specific providers
- Examples: `theta`, `openai`, `anthropic`, `google-vertex`
- Useful for excluding all models using a particular provider backend

### `--resume TIMESTAMP_DIR`

- Resume evaluation from a specific timestamp directory (e.g., `results/20250922_122904`)
- Automatically skips completed models and continues with incomplete ones
- For partially completed models, resumes from failed/missing questions
- Uses existing directory structure and configuration
- Perfect for recovering from interruptions (network issues, system restarts)
- Must be used with the same evaluation mode as the original run

### `--fix TIMESTAMP_DIR`

- Fix both evaluation and judge failures from a previous run using concurrent processing
- Reads model list from `available_models_*.json`
- Processes multiple models simultaneously (up to `max_concurrent_models`)
- For each model, fixes failed questions concurrently (up to `num_workers`)
- Retries questions with empty responses in prediction files
- Retries judgments with empty judge responses in judge files
- Updates files with successful results and recalculates metrics
- Follows the same evaluation → judge → metrics workflow as normal runs

### `--moderate-calculate`

- Enable moderate calculation mode for metrics computation
- **Strict mode** (default): Uses total questions as denominator, excludes models with any incomplete judgments
- **Moderate mode**: Uses successful judgments as denominator, includes models with partial judgment failures
- Useful for maximizing data utilization when some models have minor judgment failures
- Example: Model with 2157/2158 successful judgments gets included instead of excluded

### `--print-streaming`

- Print streaming responses to console in real-time
- Useful for monitoring model responses during evaluation
- Bypasses normal logging to show immediate output

## Output Files

Results are automatically organized with timestamps for each evaluation run:

```text
results/
├── 20250922_093840/              # Timestamped run directory
│   ├── predictions/              # Model prediction results
│   │   ├── hle_openai_gpt-4o_openai_20250922_093840.json
│   │   └── hle_anthropic_claude-3.5-sonnet_anthropic_20250922_093840.json
│   ├── judged/                   # Judging results and scores
│   │   ├── judged_hle_openai_gpt-4o_openai_20250922_093840.json
│   │   └── judged_hle_anthropic_claude-3.5-sonnet_anthropic_20250922_093840.json
│   ├── question_ids_20250922_093840.json           # Question IDs used in this run
│   ├── available_models_20250922_093840.json       # Available models list
│   ├── metrics_summary_20250922_093840.json        # Aggregated metrics and results
│   ├── evaluation_statistics_20250922_093840.json  # Evaluation completion statistics
│   ├── judge_statistics_20250922_093840.json       # Judge completion statistics
│   └── metrics_statistics_20250922_093840.json     # Metrics calculation statistics
└── logs/
    └── 20250922_093840/          # Timestamped logs directory
        ├── zenmux_benchmark.log  # Main benchmark log
        └── errors.log            # Error-only log with generation IDs
```

### Statistics Files

The framework generates comprehensive statistics files with failure tracking:

- **evaluation_statistics_*.json**: Tracks which models completed evaluation successfully
- **judge_statistics_*.json**: Tracks which models completed judging successfully
- **metrics_statistics_*.json**: Tracks which models were included/excluded from metrics

Each statistics file includes a `failure_lists` section that explicitly lists failed models for easy identification and debugging.

## Concurrency Configuration

The framework uses a dual-layer concurrency architecture:

```python
# config.py settings
class HLEConfig:
    num_workers: int = 3              # Inner concurrency: requests per model
    max_concurrent_models: int = 60   # Outer concurrency: simultaneous models
```

- **Outer Layer** (`max_concurrent_models`): Multiple models evaluated simultaneously
- **Inner Layer** (`num_workers`): Concurrent requests per model
- **Example**: `max_concurrent_models=5, num_workers=10` = up to 50 total concurrent API calls

## Architecture

The project uses a modular architecture:

- `hle/` - HLE evaluation framework
  - `runner.py` - Main orchestrator with dual-layer concurrency
  - `evaluation.py` - Individual model evaluation logic with resumable execution
  - `judge.py` - Automated scoring using judge models with structured response parsing
  - `dataset.py` - HLE dataset loading and question formatting
  - `statistics.py` - Comprehensive statistics generation with failure tracking
- `zenmux/` - ZenMux API integration
  - `api.py` - ZenMux API integration and model discovery
  - `client.py` - HTTP client with connection pooling
  - `models.py` - Data models for API responses and configurations
- `config.py` - Configuration management
- `benchmark.py` - Main CLI orchestrator
- `utils/` - Logging and utility functions

## Error Handling and Logging

The framework provides comprehensive error handling and logging:

- **Structured Logging**: Separate logs for different components with appropriate log levels
- **Error Tracking**: Detailed error logs with generation IDs for debugging
- **Failure Recovery**: Automatic retry mechanisms with exponential backoff
- **Statistics Tracking**: Complete tracking of evaluation, judging, and metrics failures
- **Performance Metrics**: Detailed timing and throughput measurements

## Contributing

Contributions are welcome! Please ensure your code follows the existing patterns and includes appropriate error handling. Key areas for contribution:

- Additional judge models and scoring methods
- New evaluation datasets beyond HLE
- Performance optimizations for large-scale evaluations
- Enhanced error recovery mechanisms

## License

This project is open-source and available under the terms specified in the repository.