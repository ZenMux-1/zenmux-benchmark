# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `zenmux-benchmark`, an open-source AI model performance evaluation framework by ZenMux. The project is in early development with a minimal Python setup using uv for dependency management.

## Development Environment

- **Python Version**: 3.13+ (specified in pyproject.toml)
- **Dependency Management**: Uses `uv` instead of traditional pip/venv
- **Virtual Environment**: Located in `.venv/` (managed by uv)

## Common Commands

### Running the Application
```bash
uv run python main.py
```

### Activating Virtual Environment (if needed)
```bash
source .venv/bin/activate
```

### Installing Dependencies
```bash
uv sync
```

### Adding New Dependencies
```bash
uv add <package-name>
```

## Architecture

The project currently has a minimal structure:
- `main.py`: Entry point with basic "Hello World" functionality
- `pyproject.toml`: Project configuration with OpenAI dependency
- Uses uv for modern Python dependency management

## Key Dependencies

- `openai>=1.107.3`: OpenAI API client (likely for AI model evaluation tasks)

## Development Notes

- This project uses `uv` instead of traditional pip/virtualenv workflow
- Always use `uv run` prefix when executing Python scripts to ensure proper environment
- The project is designed to be an AI model evaluation framework but is currently in early setup phase