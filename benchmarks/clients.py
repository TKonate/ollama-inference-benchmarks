"""Backend client factory and OpenAI-compatible client (llama.cpp, vLLM).

Both llama.cpp (``llama-server``) and vLLM expose OpenAI-compatible
endpoints; a single client covers them. Ollama keeps its native client.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from benchmarks.client import OllamaClient
from benchmarks.models import BackendType, BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)


class BackendClient(Protocol):
    """Common interface implemented by every inference backend client."""

    def generate(self, request: BenchmarkRequest) -> BenchmarkResult: ...

    def health_check(self) -> bool: ...

    def list_models(self) -> list[str]: ...


def build_chat_payload(request: BenchmarkRequest) -> dict[str, Any]:
    """Build the OpenAI-style /v1/chat/completions request body."""
    return {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": False,
    }


def parse_chat_response(
    data: dict[str, Any],
    request: BenchmarkRequest,
    elapsed_seconds: float,
) -> BenchmarkResult:
    """Convert an OpenAI-compatible completion response into a BenchmarkResult."""
    content = ""
    with contextlib.suppress(KeyError, IndexError, TypeError):
        content = data["choices"][0]["message"]["content"]

    usage: dict[str, Any] = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    result = BenchmarkResult(
        model=request.model,
        prompt=request.prompt,
        response_text=content or "",
        elapsed_seconds=round(elapsed_seconds, 2),
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
    )
    result.compute_throughput()
    return result


class OpenAICompatClient:
    """Client for OpenAI-compatible servers (llama.cpp, vLLM).

    Point ``base_url`` at the server root, e.g. http://127.0.0.1:8080.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Send a chat completion request and return a BenchmarkResult."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = json.dumps(build_chat_payload(request)).encode()

        http_request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
                data: dict[str, Any] = json.load(response)
        except urllib.error.URLError as exc:
            elapsed = time.perf_counter() - started
            return BenchmarkResult(
                model=request.model,
                prompt=request.prompt,
                elapsed_seconds=round(elapsed, 2),
                success=False,
                error=str(exc),
            )
        except TimeoutError as exc:
            elapsed = time.perf_counter() - started
            return BenchmarkResult(
                model=request.model,
                prompt=request.prompt,
                elapsed_seconds=round(elapsed, 2),
                success=False,
                error=f"Timeout after {request.timeout}s: {exc}",
            )

        elapsed = time.perf_counter() - started
        return parse_chat_response(data, request, elapsed)

    def health_check(self) -> bool:
        """Return True if the server exposes /v1/models."""
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                status: int = resp.status
                return status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return the model ids the server advertises."""
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data: dict[str, Any] = json.load(resp)
            return [str(m["id"]) for m in data.get("data", [])]
        except Exception as exc:
            logger.warning("Failed to list models from %s: %s", self.base_url, exc)
            return []


def get_client(backend: BackendType, base_url: str, timeout: int = 600) -> BackendClient:
    """Instantiate the client matching the requested backend."""
    if backend == BackendType.OLLAMA:
        return OllamaClient(base_url=base_url, timeout=timeout)
    if backend in (BackendType.LLAMACPP, BackendType.VLLM):
        return OpenAICompatClient(base_url=base_url, timeout=timeout)
    raise ValueError(f"Unsupported backend: {backend!r}")
