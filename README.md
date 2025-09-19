# ZenMux Benchmark

The open-source AI model performance evaluation framework by ZenMux. Official benchmark suite for evaluating AI model performance across multiple dimensions.

## Overview

ZenMux Benchmark is a production-grade evaluation framework that enables comprehensive testing of AI models on the Humanity's Last Exam (HLE) dataset through the ZenMux platform. It provides seamless integration with ZenMux's unified API to evaluate models from multiple providers including OpenAI, Anthropic, Google, and DeepSeek.

### Key Features

- 🌐 **Unified API Access**: Evaluate models from multiple providers through ZenMux's single API
- 🧠 **HLE Integration**: Built-in support for Humanity's Last Exam, a frontier AI benchmark
- 🔄 **Automatic Judging**: Intelligent scoring system using advanced judge models
- 📊 **Text-Only Mode**: Filter multimodal questions for text-only model evaluation
- 🚫 **Smart Model Exclusion**: Flexible exclusion system supporting exact matches, vendor filtering, and model name patterns
- ⚡ **Dual-Layer Concurrency**: Advanced parallel processing with model-level and request-level concurrency
- 🔧 **Failure Recovery**: Intelligent fix system for recovering from evaluation and judge failures
- ⚙️ **Production Ready**: Robust error handling, retry mechanisms, and resumable evaluations
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

# Small-scale test - text-only questions, 5 samples
uv run python benchmark.py --text-only --max-samples 3

# Test specific model
uv run python benchmark.py --mode single \
  --model-slug  openai/gpt-5-nano\
   --provider-slug openai \
  --text-only --max-samples 10 \

uv run python benchmark.py --mode single \
  --model-slug openai/gpt-4o-mini \
   --provider-slug openai \
  --text-only --max-samples 10 \
  --no-judge


# Test all models except expensive ones
uv run python benchmark.py --text-only --max-samples 5 --exclude openai/gpt-4o
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
# Evaluate all GPT models
uv run python benchmark.py --mode filter --model-filter deepseek --text-only

# Evaluate all Claude models, text-only questions
uv run python benchmark.py --mode filter --model-filter claude --text-only

# Evaluate all models from OpenAI provider
uv run python benchmark.py --mode filter --model-filter openai
```

### 4. Model Exclusion

```bash
# Exclude specific models (all providers)
uv run python benchmark.py --exclude openai/gpt-4o anthropic/claude-3-haiku

# Exclude entire vendors
uv run python benchmark.py --exclude anthropic openai

# Exclude by model name only (affects all providers)
uv run python benchmark.py --exclude gpt-4o

# Combine with other options
uv run python benchmark.py --mode filter --model-filter gpt --exclude openai/gpt-4o-mini
```

### 5. Failure Recovery

When evaluations encounter failures, you can automatically retry and fix them:

```bash
# Fix both evaluation and judge failures from a previous run
uv run python benchmark.py --fix results/20250919_011623

# The system will:
# - Read models from available_models_*.json
# - Check prediction files for questions with has_answer=false
# - Check judge files for questions with has_judgment=false
# - Retry failed questions and judgments automatically
# - Update files with successful results
# - Recalculate metrics for all models
```

## Important Options

### `--text-only`

- Filters out questions containing images, evaluates text-only questions
- Suitable for models that don't support multimodal inputs
- Significantly reduces evaluation time

### `--max-samples N`

- Limits evaluation to the first N questions
- Perfect for testing and debugging
- Recommended to start with small values (5-10)

### `--no-judge`

- Skips automatic judging
- Use when you want to manually score results

### `--num-workers N`

- Controls concurrent requests per model (inner concurrency)
- Default is 2, adjust based on API rate limits and provider capabilities
- Each model processes questions concurrently with this limit

### `--exclude MODEL1 MODEL2 ...`

- Exclude specific models from evaluation
- Supports three matching patterns:
  - **Exact match**: `openai/gpt-4o` (excludes only this specific model)
  - **Vendor exclusion**: `anthropic` (excludes all Anthropic models)
  - **Model name**: `gpt-4o` (excludes gpt-4o from all providers)
- Can combine multiple exclusion patterns

### `--fix TIMESTAMP_DIR`

- Fix both evaluation and judge failures from a previous run
- Reads model list from `available_models_*.json`
- Retries questions with `has_answer=false` in prediction files
- Retries judgments with `has_judgment=false` in judge files
- Updates files with successful results and recalculates metrics

## Output Files

Results are automatically organized with timestamps for each evaluation run:

```text
results/
├── 20250917_173456/              # Timestamped run directory
│   ├── predictions/              # Model prediction results (with has_answer field)
│   │   ├── hle_openai_gpt-4o_openai_20250917_173456.json
│   │   └── hle_anthropic_claude-3.5-sonnet_anthropic_20250917_173456.json
│   ├── judged/                   # Judging results and scores (with has_judgment field)
│   │   ├── judged_hle_openai_gpt-4o_openai_20250917_173456.json
│   │   └── judged_hle_anthropic_claude-3.5-sonnet_anthropic_20250917_173456.json
│   ├── question_ids_20250917_173456.json           # Question IDs used in this run
│   ├── available_models_20250917_173456.json       # Available models list
│   └── metrics_summary_20250917_173456.json        # Aggregated metrics and results
└── 20250917_180234/              # Another evaluation run
    ├── predictions/
    ├── judged/
    ├── question_ids_20250917_180234.json
    ├── available_models_20250917_180234.json
    └── metrics_summary_20250917_180234.json
```

## Performance & Concurrency

ZenMux Benchmark implements a **dual-layer concurrency architecture** for maximum performance:

### Dual-Layer Concurrency

The system operates with two independent levels of concurrency:

1. **Model-level Concurrency (Outer Layer)**

   - Multiple models evaluated simultaneously
   - Controlled by `max_concurrent_models` configuration (default: 3)
   - Example: GPT-4, Claude, and Gemini running in parallel

2. **Request-level Concurrency (Inner Layer)**
   - Multiple questions per model processed concurrently
   - Controlled by `num_workers` configuration (default: 2)
   - Example: Each model processes 2 questions simultaneously

### Performance Benefits

**Example Scenario**: 6 models × 1000 questions each

- **Serial execution**: ~17 minutes (1 model at a time)
- **Dual-layer concurrent**: ~6 minutes (3 models in parallel)
- **Performance gain**: 3x faster ⚡

### Configuration Tuning

Edit `config.py` or create custom configurations:

```python
class HLEConfig:
    num_workers: int = 2               # Inner: requests per model
    max_concurrent_models: int = 3     # Outer: simultaneous models
```

**Conservative Settings** (avoid rate limits):

```python
max_concurrent_models = 1
num_workers = 1
```

**Aggressive Settings** (maximum speed, stable network):

```python
max_concurrent_models = 5
num_workers = 5
```

### Rate Limit Considerations

- **ZenMux API**: Adjust `max_concurrent_models` based on your plan
- **Provider Limits**: Some providers have stricter per-model limits
- **Network Stability**: Higher concurrency requires stable connections

## Usage Examples

### Development and Testing

```bash
# Quick functionality test
uv run python example.py

# Small-scale evaluation test
uv run python benchmark.py --text-only --max-samples 3

# Test specific model
uv run python benchmark.py --mode single \
  --model-slug openai/gpt-4o-mini --provider-slug openai \
  --max-samples 5
```

### Production Evaluation

```bash
# Evaluate all text models (recommended)
uv run python benchmark.py --mode all --text-only

# Evaluate all models except expensive ones
uv run python benchmark.py --mode all --exclude openai/gpt-4o anthropic/claude-opus-4.1

# Evaluate all models from specific provider, excluding problematic models
uv run python benchmark.py --mode filter --model-filter anthropic --exclude anthropic/claude-3-haiku

# Skip all OpenAI models (cost control)
uv run python benchmark.py --mode all --exclude openai --text-only
```

### CI/CD Integration

```bash
# Commands suitable for CI/CD - focused evaluation
uv run python benchmark.py --mode filter \
  --model-filter gpt \
  --exclude openai/gpt-4o \
  --text-only \
  --max-samples 50 \
  --num-workers 5

# Cost-controlled evaluation for automated testing
uv run python benchmark.py --mode all \
  --exclude openai anthropic/claude-opus-4.1 \
  --text-only \
  --max-samples 20

# Fix failures from CI/CD runs
uv run python benchmark.py --fix results/20250917_173456
```

## Important Notes

1. **Dual-Layer Concurrency**: The system runs multiple models simultaneously (outer layer) with concurrent requests per model (inner layer)
2. **API Rate Limits**: Configure both `max_concurrent_models` and `num_workers` based on your ZenMux plan and provider limits
3. **Cost Control**: Use `--max-samples` and `--exclude` to control evaluation scope and costs - exclude expensive models to save budget
4. **Model Exclusion**: Use `--exclude` strategically to skip problematic, expensive, or irrelevant models for your use case
5. **Failure Recovery**: The system tracks failures using `has_answer` and `has_judgment` fields, use `--fix` to retry failed operations
6. **Network Stability**: Higher concurrency requires stable network connections for optimal performance
7. **Storage Space**: Full evaluations generate large amounts of data, timestamped directories help organize results by run
8. **Judge Model**: Default uses `openai/gpt-5:openai`, judging also benefits from concurrent processing
9. **Memory Usage**: Monitor system memory with high concurrency settings

## Troubleshooting

### Common Errors

1. **`ZENMUX_API_KEY not set`**

   - Ensure environment variable is set: `export ZENMUX_API_KEY="your_key"`

2. **`504 Gateway Time-out`**

   - Network timeout, system will automatically retry
   - Consider reducing `--num-workers` value

3. **Judge model 404 error**

   - ZenMux API key issue or judge model unavailable
   - Use `--no-judge` to skip judging

4. **Memory issues**

   - Reduce `--num-workers` value
   - Use `--max-samples` to limit sample size

5. **Evaluation or judge failures**

   - Use `--fix` to retry failed operations (both evaluation and judge)
   - Check prediction/judge files for questions with `has_answer=false` or `has_judgment=false`
   - Consider reducing concurrency if failures persist

6. **Too many open files error**
   - Reduce `max_concurrent_models` and `num_workers` values
   - This error occurs when file handle limits are exceeded during high concurrency operations

### Performance Optimization

**Dual-Layer Concurrency**:

- **Model-level**: Adjust `max_concurrent_models` (1-5) based on API limits
- **Request-level**: Tune `num_workers` (1-5) per provider capabilities
- Monitor system resources and network stability to avoid "too many open files" errors

**General Optimizations**:

- Use `--text-only` for 40-60% speed improvements
- Start with `--max-samples` for testing before full evaluation
- Balance concurrency vs. stability based on network conditions

**Recommended Settings by Use Case**:

```bash
# Development/Testing (safe)
max_concurrent_models = 1, num_workers = 1

# Production/CI (balanced)
max_concurrent_models = 3, num_workers = 2

# High-performance (stable network, careful monitoring)
max_concurrent_models = 5, num_workers = 3
```

## GitHub Actions

The repository includes a GitHub Actions workflow for automated benchmarking. The workflow is configured for manual triggering and supports customizable parameters:

- Model filtering
- Text-only mode
- Sample limits
- Worker count

## Architecture

The project uses a modular architecture:

- `hle/` - HLE evaluation framework
- `zenmux/` - ZenMux API integration
- `config.py` - Configuration management
- `benchmark.py` - Main orchestrator

## Contributing

Contributions are welcome! Please ensure your code follows the existing patterns and includes appropriate error handling.

## License

This project is open-source and available under the terms specified in the repository.
