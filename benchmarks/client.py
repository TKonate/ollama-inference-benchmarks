"""Ollama API client with retry logic, streaming, and error handling."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from benchmarks.models import BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)

_NS_PER_SECOND = 1_000_000_000


class OllamaClientError(Exception):
    """Raised when the Ollama API request fails."""


def _seconds(nanoseconds: object) -> float | None:
    """Convert an Ollama duration in nanoseconds to seconds."""
    if isinstance(nanoseconds, int | float) and nanoseconds > 0:
        return round(float(nanoseconds) / _NS_PER_SECOND, 4)
    return None


def _apply_ollama_timings(result: BenchmarkResult, data: dict[str, Any]) -> None:
    """Copy Ollama's own timing counters onto a result.

    Ollama reports model load, prompt evaluation and generation separately.
    Keeping them apart is what makes cold vs. warm and prompt-processing cost
    separable in the published dataset.
    """
    result.load_seconds = _seconds(data.get("load_duration"))
    result.prompt_eval_seconds = _seconds(data.get("prompt_eval_duration"))
    result.eval_seconds = _seconds(data.get("eval_duration"))

    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    if isinstance(prompt_tokens, int):
        result.prompt_tokens = prompt_tokens
    if isinstance(completion_tokens, int):
        result.completion_tokens = completion_tokens
    result.compute_throughput()


class OllamaClient:
    """Minimal client for the Ollama /api/generate endpoint.

    ``generate`` supports both response modes: non-streamed (a single JSON body)
    and streamed (newline-delimited JSON chunks). Streaming is what makes
    time-to-first-token measurable, so benchmark campaigns always use it.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[str]:
        """Return the tags Ollama advertises locally."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data: dict[str, Any] = json.load(resp)
            return [str(m["name"]) for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("Failed to list models from %s: %s", self.base_url, exc)
            return []

    def model_details(self, model: str) -> dict[str, Any]:
        """Return the /api/show metadata block for a model (empty on failure)."""
        payload = json.dumps({"model": model}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data: dict[str, Any] = json.load(response)
            return data
        except Exception as exc:
            logger.warning("Failed to read details for %s: %s", model, exc)
            return {}

    def running_models(self) -> list[dict[str, Any]]:
        """Return the models currently loaded in memory (/api/ps)."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/ps")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data: dict[str, Any] = json.load(resp)
            models: list[dict[str, Any]] = data.get("models", [])
            return models
        except Exception as exc:
            logger.warning("Failed to read /api/ps: %s", exc)
            return []

    def version(self) -> str | None:
        """Return the server version string."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/version")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data: dict[str, Any] = json.load(resp)
            version = data.get("version")
            return str(version) if version else None
        except Exception:
            return None

    def unload(self, model: str) -> bool:
        """Evict a model so the next run measures a real cold start."""
        payload = json.dumps({"model": model, "keep_alive": 0}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()
            return True
        except Exception as exc:
            logger.warning("Failed to unload %s: %s", model, exc)
            return False

    def wait_until_unloaded(self, model: str, timeout: float = 60.0) -> bool:
        """Poll /api/ps until no instance of the model is resident."""
        deadline = time.monotonic() + timeout
        family = model.split(":")[0]
        while time.monotonic() < deadline:
            loaded = {str(m.get("name", "")).split(":")[0] for m in self.running_models()}
            if family not in loaded:
                return True
            time.sleep(0.25)
        return False

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Send a generate request and return a BenchmarkResult."""
        if request.stream:
            return self._generate_stream(request)
        return self._generate_once(request)

    def _build_payload(self, request: BenchmarkRequest, *, stream: bool) -> bytes:
        body: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": stream,
        }
        if request.keep_alive is not None:
            body["keep_alive"] = request.keep_alive

        options = dict(request.options)
        # "think" is a top-level API parameter, not a sampling option: keeping it
        # inside "options" makes Ollama ignore it silently.
        think = options.pop("think", None)
        if think is not None:
            body["think"] = think
        if options:
            body["options"] = options
        return json.dumps(body).encode()

    def _new_result(self, request: BenchmarkRequest) -> BenchmarkResult:
        return BenchmarkResult(
            model=request.model,
            prompt=request.prompt,
            backend=request.backend,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def _generate_once(self, request: BenchmarkRequest) -> BenchmarkResult:
        http_request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=self._build_payload(request, stream=False),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
                data: dict[str, Any] = json.load(response)
        except urllib.error.URLError as exc:
            return self._failure(request, time.perf_counter() - started, str(exc))
        except TimeoutError as exc:
            return self._failure(
                request, time.perf_counter() - started, f"Timeout after {request.timeout}s: {exc}"
            )

        elapsed = time.perf_counter() - started
        result = self._new_result(request)
        result.response_text = str(data.get("response", ""))
        result.elapsed_seconds = round(elapsed, 3)
        _apply_ollama_timings(result, data)
        return result

    def _generate_stream(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Stream a response, recording time-to-first-token as chunks arrive."""
        http_request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=self._build_payload(request, stream=True),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        chunks: list[str] = []
        ttft_ms: float | None = None
        final: dict[str, Any] = {}

        try:
            with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
                for raw_line in response:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON stream line: %r", line[:80])
                        continue
                    if chunk.get("error"):
                        return self._failure(
                            request, time.perf_counter() - started, str(chunk["error"])
                        )
                    piece = chunk.get("response")
                    if isinstance(piece, str):
                        if piece and ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - started) * 1000, 1)
                        chunks.append(piece)
                    if chunk.get("done"):
                        final = chunk
                        break
        except urllib.error.URLError as exc:
            return self._failure(request, time.perf_counter() - started, str(exc))
        except TimeoutError as exc:
            return self._failure(
                request, time.perf_counter() - started, f"Timeout after {request.timeout}s: {exc}"
            )

        elapsed = time.perf_counter() - started
        result = self._new_result(request)
        result.response_text = "".join(chunks)
        result.elapsed_seconds = round(elapsed, 3)
        result.ttft_ms = ttft_ms
        _apply_ollama_timings(result, final)
        return result

    def _failure(self, request: BenchmarkRequest, elapsed: float, message: str) -> BenchmarkResult:
        result = self._new_result(request)
        result.success = False
        result.error = message
        result.elapsed_seconds = round(elapsed, 3)
        return result

    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                status: int = resp.status
                return status == 200
        except Exception:
            return False
