"""Tests for report aggregation and rendering (bench report)."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.models import BenchmarkResult
from benchmarks.report import (
    bar_chart_svg,
    load_dataset,
    render_markdown,
    summarize,
    write_report,
)


def _result(
    model: str,
    *,
    phase: str = "warm",
    elapsed: float = 10.0,
    ttft: float | None = 200.0,
    tps: float | None = 20.0,
    success: bool = True,
    rss: float | None = 900.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        prompt="p",
        elapsed_seconds=elapsed,
        ttft_ms=ttft,
        tokens_per_second=tps,
        phase=phase,
        server_peak_rss_mib=rss,
        success=success,
        load_seconds=2.5 if phase == "cold" else None,
        parameter_size="1B",
        quantization="Q4_K_M",
    )


class TestSummarize:
    def test_cold_runs_are_excluded_from_latency(self) -> None:
        rows = [
            _result("m:1b", phase="cold", elapsed=30.0, ttft=9000.0),
            _result("m:1b", phase="warm", elapsed=10.0, ttft=200.0),
            _result("m:1b", phase="warm", elapsed=12.0, ttft=220.0),
        ]
        summary = summarize(rows)[0]
        assert summary.median_elapsed_seconds == 11.0
        assert summary.median_ttft_ms == 210.0
        assert summary.median_load_seconds == 2.5

    def test_failures_are_counted_and_excluded(self) -> None:
        rows = [
            _result("m:1b", phase="warm"),
            _result("m:1b", success=False, elapsed=0.1),
        ]
        summary = summarize(rows)[0]
        assert summary.runs == 1
        assert summary.failures == 1
        assert summary.failure_rate == 0.5

    def test_p95_uses_upper_observation(self) -> None:
        rows = [_result("m:1b", elapsed=float(value)) for value in range(1, 21)]
        summary = summarize(rows)[0]
        assert summary.p95_elapsed_seconds is not None
        assert summary.p95_elapsed_seconds >= 19.0

    def test_models_are_sorted(self) -> None:
        summaries = summarize([_result("z:1b"), _result("a:1b")])
        assert [s.model for s in summaries] == ["a:1b", "z:1b"]


class TestSvgChart:
    def test_contains_labels_and_values(self) -> None:
        svg = bar_chart_svg("Speed", {"qwen3:0.6b": 25.5, "llama3.2:1b": 10.0}, unit="tok/s")
        assert svg.startswith("<svg")
        assert "qwen3:0.6b" in svg
        assert "25.5 tok/s" in svg
        assert svg.count("<rect") == 2

    def test_escapes_markup_in_labels(self) -> None:
        svg = bar_chart_svg("Speed", {"<script>": 1.0}, unit="tok/s")
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg

    def test_empty_values_yield_empty_svg(self) -> None:
        assert "svg" in bar_chart_svg("Empty", {}, unit="tok/s")


class TestDatasetRoundTrip:
    def test_load_dataset_reads_environment_and_rows(self, tmp_path: Path) -> None:
        dataset = tmp_path / "d.jsonl"
        dataset.write_text(
            "\n".join(
                [
                    json.dumps({"type": "environment", "environment": {"cpu_model": "EPYC"}}),
                    json.dumps({"type": "result", "result": _result("m:1b").model_dump()}),
                    json.dumps({"type": "summary", "results": 1, "failures": 0}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        environment, results = load_dataset(dataset)
        assert environment is not None
        assert environment["environment"]["cpu_model"] == "EPYC"
        assert len(results) == 1
        assert results[0].model == "m:1b"

    def test_write_report_creates_markdown_and_charts(self, tmp_path: Path) -> None:
        dataset = tmp_path / "campaign.jsonl"
        dataset.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "environment",
                            "environment": {"cpu_model": "EPYC", "cpu_count": 4},
                            "options": {"temperature": 0},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "result": _result("m:1b", elapsed=8.0, ttft=150.0).model_dump(),
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        written = write_report(dataset, tmp_path / "docs")
        names = {path.name for path in written}
        assert names == {"campaign.md", "campaign-throughput.svg", "campaign-ttft.svg"}
        markdown = (tmp_path / "docs" / "campaign.md").read_text(encoding="utf-8")
        assert "EPYC" in markdown
        assert "`m:1b`" in markdown
        assert "What these numbers do not tell you" in markdown


class TestRenderMarkdown:
    def test_warns_about_quality_and_noise(self) -> None:
        markdown = render_markdown(None, summarize([_result("m:1b")]), total_runs=1)
        assert "Quality." in markdown
        assert "Noise." in markdown
        assert "| Model |" in markdown
