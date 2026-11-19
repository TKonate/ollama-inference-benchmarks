"""Request and result schema shared by every backend client.

``BenchmarkResult`` is the row that reaches the JSONL dataset and then the
report, so the units declared here (seconds, milliseconds, MiB, tokens per
second) are the units the published tables carry. This module validates and
derives; it does not measure, since every timing field is filled in by a
client.
"""

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
    stream: bool = Field(default=False, description="Stream the response to measure TTFT")
    keep_alive: str | int | None = Field(
        default=None,
        description="Ollama keep_alive value; 0 unloads the model right after the run",
    )
    options: dict[str, object] = Field(
        default_factory=dict, description="Backend-specific sampling options"
    )


class BenchmarkResult(BaseModel):
    """Output of a single benchmark run.

    Two latency families are recorded:

    * ``elapsed_seconds``: wall clock, from request send to last byte received.
      It includes queuing, model load and sampling.
    * ``ttft_ms``: time to first token, measured on streamed responses. This is
      what a user actually perceives before text starts appearing.

    Token throughput is derived from the backend's own counters
    (``eval_count``/``eval_duration`` for Ollama, ``usage`` for
    OpenAI-compatible servers) rather than from wall clock, so it measures
    generation speed and not network or load overhead.
    """

    model: str
    prompt: str
    backend: BackendType = BackendType.OLLAMA
    response_text: str = ""
    elapsed_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    peak_memory_mib: float | None = Field(
        default=None,
        description="Peak RSS of the inference server process during the run (MiB)",
    )
    success: bool = True
    error: str | None = None

    # Measurement detail
    run_index: int = Field(default=0, ge=0, description="0-based repeat index within a cell")
    # Warm by default: a client that never evicted the model cannot know whether
    # the load cost was already paid, and claiming cold would inflate latency.
    phase: str = Field(default="warm", description="'cold' when the model had to be loaded")
    category: str | None = Field(default=None, description="Task category from the suite")
    prompt_id: str | None = Field(default=None, description="Stable identifier of the prompt")
    ttft_ms: float | None = Field(default=None, description="Time to first token, milliseconds")
    load_seconds: float | None = Field(default=None, description="Backend-reported model load time")
    eval_seconds: float | None = Field(default=None, description="Backend-reported generation time")
    prompt_eval_seconds: float | None = Field(
        default=None, description="Backend-reported prompt processing time"
    )
    server_peak_rss_mib: float | None = Field(
        default=None, description="Peak RSS of the server process, sampled (MiB)"
    )
    model_size_mib: float | None = Field(
        default=None, description="Model size in memory as reported by the backend (MiB)"
    )
    parameter_size: str | None = None
    quantization: str | None = None
    digest: str | None = None
    started_at: str | None = Field(default=None, description="ISO-8601 UTC start timestamp")

    def compute_throughput(self) -> None:
        """Derive tokens per second from the most reliable timing source available.

        The backend's own generation timer comes first because it excludes model
        load and prompt processing, which is what makes a cold run comparable to
        a warm one. The wall-clock fallback is all a server without counters
        allows; it covers the whole request, so on a cold run the value is a
        lower bound on generation speed, not an estimate of it.
        """
        if not self.completion_tokens:
            # No output means no rate: leaving None keeps the report showing a
            # dash instead of a plausible-looking zero.
            return
        if self.eval_seconds and self.eval_seconds > 0:
            self.tokens_per_second = round(self.completion_tokens / self.eval_seconds, 2)
        elif self.elapsed_seconds > 0:
            self.tokens_per_second = round(self.completion_tokens / self.elapsed_seconds, 2)


class ModelProfile(BaseModel):
    """Static profile for a model being benchmarked."""

    tag: str
    parameter_count: str | None = None  # e.g. "1.7B"
    quantization: str | None = None  # e.g. "Q4_K_M"
    context_length: int | None = None
