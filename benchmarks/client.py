"""Ollama API client with retry logic and error handling."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from benchmarks.models import BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Raised when the Ollama API request fails."""


class OllamaClient:
    """Minimal client for the Ollama /api/generate endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Send a generate request and return a BenchmarkResult."""
        url = f"{self.base_url}/api/generate"
        payload = json.dumps(
            {
                "model": request.model,
                "prompt": request.prompt,
                "stream": request.stream,
            }
        ).encode()

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

        # Extract optional token counts from Ollama response
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")

        result = BenchmarkResult(
            model=request.model,
            prompt=request.prompt,
            response_text=data.get("response", ""),
            elapsed_seconds=round(elapsed, 2),
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        )
        result.compute_throughput()
        return result

    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
