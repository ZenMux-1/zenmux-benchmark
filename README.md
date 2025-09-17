# ZenMux Benchmark

The open-source AI model performance evaluation framework by ZenMux. Official benchmark suite for evaluating AI model performance across multiple dimensions.

## Overview

ZenMux Benchmark is a production-grade evaluation framework that enables comprehensive testing of AI models on the Humanity's Last Exam (HLE) dataset through the ZenMux platform. It provides seamless integration with ZenMux's unified API to evaluate models from multiple providers including OpenAI, Anthropic, Google, and DeepSeek.

### Key Features

- 🌐 **Unified API Access**: Evaluate models from multiple providers through ZenMux's single API
- 🧠 **HLE Integration**: Built-in support for Humanity's Last Exam, a frontier AI benchmark
- 🔄 **Automatic Judging**: Intelligent scoring system using advanced judge models
- 📊 **Text-Only Mode**: Filter multimodal questions for text-only model evaluation
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
uv run python benchmark.py --text-only --max-samples 5

# Test specific model
uv run python benchmark.py --mode single \
  --model-slug openai/gpt-4o-mini \
  --provider-slug openai \
  --text-only --max-samples 5
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
  --model-slug openai/gpt-4.1-mini \
  --provider-slug openai

# The system automatically evaluates all available endpoints for the model
```

### 3. Filtered Evaluation

```bash
# Evaluate all GPT models
uv run python benchmark.py --mode filter --model-filter gpt

# Evaluate all Claude models, text-only questions
uv run python benchmark.py --mode filter --model-filter claude --text-only

# Evaluate all models from OpenAI provider
uv run python benchmark.py --mode filter --model-filter openai
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

- Controls the number of concurrent evaluations
- Default is 10, adjust based on API rate limits

## Output Files

After evaluation, results are saved in the following structure:

```text
results/
├── predictions/     # Model prediction results
│   ├── hle_openai_gpt-4.1-mini_openai.json
│   └── hle_anthropic_claude-3.5-sonnet_anthropic.json
└── judged/         # Judging results and scores
    ├── judged_hle_openai_gpt-4.1-mini_openai.json
    └── judged_hle_anthropic_claude-3.5-sonnet_anthropic.json
```

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

# Evaluate all multimodal models
uv run python benchmark.py --mode all

# Evaluate all models from specific provider
uv run python benchmark.py --mode filter --model-filter anthropic
```

### CI/CD Integration

```bash
# Commands suitable for CI/CD
uv run python benchmark.py --mode filter \
  --model-filter gpt-4o \
  --text-only \
  --max-samples 50 \
  --num-workers 5
```

## Important Notes

1. **API Limits**: Be mindful of ZenMux API rate limits, adjust `--num-workers` accordingly
2. **Cost Control**: Use `--max-samples` to control evaluation scope and costs
3. **Network Stability**: Ensure stable network connection for long-running evaluations
4. **Storage Space**: Full evaluations generate large amounts of data
5. **Judge Model**: Default uses `openai/gpt-5:openai`, errors won't affect predictions

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

### Performance Optimization

- Use `--text-only` for significant speed improvements
- Optimal `--num-workers` values are typically 5-20
- Consider batch evaluation instead of evaluating all models at once

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
