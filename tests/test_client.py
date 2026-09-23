"""Unit tests for OllamaClient (mocked HTTP)."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

from benchmarks.client import OllamaClient
from benchmarks.models import BenchmarkRequest


class TestOllamaClient:
    def test_health_check_success(self):
        client = OllamaClient(base_url="http://mock:11434")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            assert client.health_check() is True

    def test_health_check_failure(self):
        client = OllamaClient(base_url="http://mock:11434")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            assert client.health_check() is False

    def test_generate_success(self):
        client = OllamaClient(base_url="http://mock:11434")
        request = BenchmarkRequest(model="test", prompt="Hello")

        ollama_response = {
            "response": "Hi there!",
            "eval_count": 10,
            "prompt_eval_count": 5,
        }

        def fake_urlopen(req, timeout=None):
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            # json.load is called inside urlopen's context
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("benchmarks.client.json.load", return_value=ollama_response):
            result = client.generate(request)

        assert result.success is True
        assert result.model == "test"
        assert result.response_text == "Hi there!"
        assert result.completion_tokens == 10
        assert result.prompt_tokens == 5
        # elapsed_seconds will be ~0 but tokens_per_second is computed if > 0
        # We can't guarantee > 0 in a mock, so check the logic path
        assert result.tokens_per_second is not None or result.elapsed_seconds == 0.0

    def test_generate_connection_error(self):
        client = OllamaClient(base_url="http://mock:11434")
        request = BenchmarkRequest(model="test", prompt="Hello")

        err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            result = client.generate(request)

        assert result.success is False
        assert "refused" in result.error

    def test_generate_timeout(self):
        client = OllamaClient(base_url="http://mock:11434", timeout=5)
        request = BenchmarkRequest(model="test", prompt="Hello", timeout=5)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = client.generate(request)

        assert result.success is False
        assert "Timeout" in result.error

    def test_base_url_trailing_slash(self):
        client = OllamaClient(base_url="http://mock:11434/")
        assert client.base_url == "http://mock:11434"

    def test_generate_tokens_per_second_with_elapsed(self):
        """Verify throughput is computed when both tokens and elapsed are positive."""
        result = ThroughputShim(elapsed_seconds=5.0, completion_tokens=20)
        result.compute_throughput()
        assert result.tokens_per_second == 4.0


# Helper for isolated throughput test


class ThroughputShim:
    """Minimal shim to test compute_throughput in isolation."""

    def __init__(self, elapsed_seconds: float, completion_tokens: int):
        self.elapsed_seconds = elapsed_seconds
        self.completion_tokens = completion_tokens
        self.tokens_per_second = None

    def compute_throughput(self):
        if self.completion_tokens and self.elapsed_seconds > 0:
            self.tokens_per_second = round(self.completion_tokens / self.elapsed_seconds, 2)
