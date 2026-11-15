"""Tests for the campaign runner (bench matrix)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import matrix as matrix_module
from benchmarks.matrix import CampaignConfig, build_cells, run_campaign
from benchmarks.models import BenchmarkResult


class FakeClient:
    """Stand-in for OllamaClient: no network, deterministic results."""

    def __init__(self, *, fail_models: set[str] | None = None) -> None:
        self.fail_models = fail_models or set()
        self.unloaded: list[str] = []
        self.requests: list[str] = []

    def health_check(self) -> bool:
        return True

    def version(self) -> str:
        return "9.9.9-fake"

    def unload(self, model: str) -> bool:
        self.unloaded.append(model)
        return True

    def wait_until_unloaded(self, model: str) -> bool:
        return True

    def model_details(self, model: str) -> dict[str, object]:
        return {
            "details": {"parameter_size": "1B", "quantization_level": "Q4_K_M"},
            "digest": f"sha-{model}",
        }

    def running_models(self) -> list[dict[str, object]]:
        return [{"name": "fake:1b", "size": 512 * 1024 * 1024}]

    def generate(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request.model)
        if request.model in self.fail_models:
            return BenchmarkResult(
                model=request.model, prompt=request.prompt, success=False, error="boom"
            )
        return BenchmarkResult(
            model=request.model,
            prompt=request.prompt,
            response_text="ok",
            elapsed_seconds=1.5,
            ttft_ms=120.0,
            eval_seconds=1.0,
            completion_tokens=30,
            tokens_per_second=30.0,
        )


def test_build_cells_respects_prompts_per_category() -> None:
    config = CampaignConfig(models=["a:1b"], prompts_per_category=2)
    cells = build_cells(config)
    categories = {cell.category for cell in cells}
    assert categories == {"summarization", "classification", "code_generation"}
    assert len(cells) == 6  # 3 categories x 2 prompts
    assert len({cell.prompt_id for cell in cells}) == 6


def test_campaign_writes_environment_results_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeClient()
    monkeypatch.setattr(matrix_module, "OllamaClient", lambda **_: fake)
    monkeypatch.setattr(matrix_module, "ServerMemoryTracker", lambda pid: _NoopTracker())

    out = tmp_path / "campaign.jsonl"
    config = CampaignConfig(models=["fake:1b"], runs=2, prompts_per_category=1)
    outcome = run_campaign(config, out)

    assert outcome["results"] == 6  # 3 categories x 2 runs
    assert outcome["failures"] == 0

    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert records[0]["type"] == "environment"
    assert records[0]["environment"]["ollama_version"] == "9.9.9-fake"
    assert records[-1]["type"] == "summary"
    results = [r for r in records if r["type"] == "result"]
    assert len(results) == 6
    # First run of the model is cold, the second is warm.
    assert results[0]["result"]["phase"] == "cold"
    assert results[1]["result"]["phase"] == "warm"
    assert results[0]["result"]["parameter_size"] == "1B"
    assert fake.unloaded == ["fake:1b"]


def test_campaign_counts_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient(fail_models={"broken:1b"})
    monkeypatch.setattr(matrix_module, "OllamaClient", lambda **_: fake)
    monkeypatch.setattr(matrix_module, "ServerMemoryTracker", lambda pid: _NoopTracker())

    out = tmp_path / "campaign.jsonl"
    config = CampaignConfig(models=["broken:1b"], runs=1, prompts_per_category=1)
    outcome = run_campaign(config, out)

    assert outcome["failures"] == 3
    assert outcome["results"] == 3


def test_campaign_raises_when_server_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeClient()
    fake.health_check = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr(matrix_module, "OllamaClient", lambda **_: fake)

    with pytest.raises(RuntimeError):
        run_campaign(CampaignConfig(models=["x:1b"]), tmp_path / "x.jsonl")


class _NoopTracker:
    def __init__(self, pid: int | None = None) -> None:
        self.pid = pid

    def start(self) -> None:
        return None

    def stop(self) -> float | None:
        return 512.0
