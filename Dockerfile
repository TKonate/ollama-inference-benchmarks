# Ollama Inference Benchmarks — Reproducible Environment
#
# Usage:
#   docker build -t ollama-bench .
#   docker run --rm ollama-bench bench run --model qwen3:1.7b --prompt "Hello"
#
# To connect to a host Ollama instance:
#   docker run --rm --network host ollama-bench bench run --model qwen3:1.7b --prompt "Hello"
#
# To run tests:
#   docker run --rm ollama-bench pytest -v

FROM python:3.12-slim AS base

LABEL maintainer="TKonate"
LABEL description="LLM benchmarks for hardware you already own — no GPU required."

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY benchmarks/ benchmarks/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/

# Install the package with all dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Health check: verify the CLI loads
RUN python -c "from benchmarks.cli import app; print('✓ CLI loaded')"

# Default: run the test suite
CMD ["pytest", "-v"]
