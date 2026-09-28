"""Unit tests for benchmark models."""

from benchmarks.models import BackendType, BenchmarkRequest, BenchmarkResult


class TestBenchmarkRequest:
    def test_defaults(self):
        req = BenchmarkRequest(model="qwen3:1.7b", prompt="Hello")
        assert req.model == "qwen3:1.7b"
        assert req.backend == BackendType.OLLAMA
        assert req.base_url == "http://127.0.0.1:11434"
        assert req.timeout == 600
        assert req.stream is False

    def test_custom_url(self):
        req = BenchmarkRequest(
            model="llama3:8b",
            prompt="Test",
            base_url="http://10.0.0.1:11434",
        )
        assert req.base_url == "http://10.0.0.1:11434"

    def test_timeout_validation(self):
        import pytest

        with pytest.raises((ValueError, Exception)):
            BenchmarkRequest(model="m", prompt="p", timeout=0)


class TestBenchmarkResult:
    def test_compute_throughput(self):
        result = BenchmarkResult(
            model="test",
            prompt="test",
            elapsed_seconds=10.0,
            completion_tokens=50,
        )
        result.compute_throughput()
        assert result.tokens_per_second == 5.0

    def test_compute_throughput_no_tokens(self):
        result = BenchmarkResult(model="test", prompt="test", elapsed_seconds=10.0)
        result.compute_throughput()
        assert result.tokens_per_second is None

    def test_success_default(self):
        result = BenchmarkResult(model="test", prompt="test")
        assert result.success is True
        assert result.error is None

    def test_failure(self):
        result = BenchmarkResult(
            model="test",
            prompt="test",
            success=False,
            error="Connection refused",
        )
        assert result.success is False
        assert result.error == "Connection refused"

    def test_serialization(self):
        result = BenchmarkResult(
            model="qwen3:1.7b",
            prompt="Hello",
            elapsed_seconds=42.5,
            tokens_per_second=3.2,
        )
        data = result.model_dump()
        assert data["model"] == "qwen3:1.7b"
        assert data["elapsed_seconds"] == 42.5


class TestBackendType:
    def test_ollama(self):
        assert BackendType.OLLAMA == "ollama"

    def test_llamacpp(self):
        assert BackendType.LLAMACPP == "llamacpp"
