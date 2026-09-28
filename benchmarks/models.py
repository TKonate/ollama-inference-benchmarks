"""Pydantic models for benchmark requests, responses, and results."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BackendType(str, Enum):
    """Supported inference backends."""

    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"  # llama.cpp llama-server (OpenAI-compatible API)
    VLLM = "vllm"  # vLLM in OpenAI-compatible mode


class BenchmarkRequest(BaseModel):
    """A single benchmark run configuration."""

    model: str = Field(..., description="Model tag (e.g. 'qwen3:1.7b')")
    prompt: str = Field(..., description="Prompt to send to the model")
    backend: BackendType = BackendType.OLLAMA
    base_url: str = Field(default="http://127.0.0.1:11434", description="API base URL")
    timeout: int = Field(default=600, ge=1, description="Request timeout in seconds")
    stream: bool = Field(default=False, description="Enable streaming (currently unused)")

    # Future: warmup_runs, repeat_count, etc.


class BenchmarkResult(BaseModel):
    """Output of a single benchmark run."""

    model: str
    prompt: str
    response_text: str = ""
    elapsed_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    peak_memory_mib: float | None = None
    success: bool = True
    error: str | None = None

    def compute_throughput(self) -> None:
        """Derive tokens/sec from completion tokens and elapsed time."""
        if self.completion_tokens and self.elapsed_seconds > 0:
            self.tokens_per_second = round(self.completion_tokens / self.elapsed_seconds, 2)


class ModelProfile(BaseModel):
    """Static profile for a model being benchmarked."""

    tag: str
    parameter_count: str | None = None  # e.g. "1.7B"
    quantization: str | None = None  # e.g. "Q4_K_M"
    context_length: int | None = None
