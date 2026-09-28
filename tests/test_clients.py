"""Tests for the backend client factory and OpenAI-compatible helpers."""

from __future__ import annotations

import pytest

from benchmarks.client import OllamaClient
from benchmarks.clients import (
    OpenAICompatClient,
    build_chat_payload,
    get_client,
    parse_chat_response,
)
from benchmarks.models import BackendType, BenchmarkRequest, BenchmarkResult


def test_factory_returns_ollama_client() -> None:
    client = get_client(BackendType.OLLAMA, "http://127.0.0.1:11434")
    assert isinstance(client, OllamaClient)


def test_factory_returns_openai_compat_for_llamacpp() -> None:
    client = get_client(BackendType.LLAMACPP, "http://127.0.0.1:8080")
    assert isinstance(client, OpenAICompatClient)


def test_factory_returns_openai_compat_for_vllm() -> None:
    client = get_client(BackendType.VLLM, "http://127.0.0.1:8000")
    assert isinstance(client, OpenAICompatClient)


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        get_client("unknown", "http://127.0.0.1:11434")  # type: ignore[arg-type]


def test_chat_payload_shape() -> None:
    request = BenchmarkRequest(model="qwen3:1.7b", prompt="Hello", timeout=30)
    payload = build_chat_payload(request)
    assert payload["model"] == "qwen3:1.7b"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["stream"] is False


def test_parse_chat_response_extracts_usage() -> None:
    request = BenchmarkRequest(model="m", prompt="p")
    data = {
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    result = parse_chat_response(data, request, 1.5)
    assert isinstance(result, BenchmarkResult)
    assert result.success
    assert result.response_text == "hello world"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert result.tokens_per_second == 1.33  # 2 / 1.5 = 1.33…
    assert result.elapsed_seconds == 1.5


def test_parse_chat_response_tolerates_missing_usage() -> None:
    request = BenchmarkRequest(model="m", prompt="p")
    result = parse_chat_response({"choices": [{"message": {}}]}, request, 1.0)
    assert result.success
    assert result.response_text == ""
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
