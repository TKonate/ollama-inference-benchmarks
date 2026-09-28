"""Tests for cold/warm latency analysis."""

from __future__ import annotations

from benchmarks.analysis import summarize_cold_warm
from benchmarks.models import BenchmarkResult


def _result(elapsed: float, *, success: bool = True) -> BenchmarkResult:
    return BenchmarkResult(model="m", prompt="p", elapsed_seconds=elapsed, success=success)


def test_empty_results() -> None:
    stats = summarize_cold_warm([])
    assert stats.cold_seconds is None
    assert stats.warm_seconds is None
    assert stats.warm_runs == 0


def test_single_run_is_cold_only() -> None:
    stats = summarize_cold_warm([_result(5.2)])
    assert stats.cold_seconds == 5.2
    assert stats.warm_seconds is None
    assert stats.warm_runs == 0


def test_cold_and_warm_median() -> None:
    stats = summarize_cold_warm([_result(10.0), _result(2.0), _result(4.0), _result(6.0)])
    assert stats.cold_seconds == 10.0
    assert stats.warm_seconds == 4.0  # median of [2, 4, 6]
    assert stats.warm_runs == 3
    assert stats.warm_over_cold == 0.4


def test_failed_runs_excluded() -> None:
    stats = summarize_cold_warm([_result(10.0), _result(3.0, success=False), _result(5.0)])
    assert stats.cold_seconds == 10.0
    assert stats.warm_seconds == 5.0
    assert stats.warm_runs == 1


def test_all_failed() -> None:
    stats = summarize_cold_warm([_result(3.0, success=False)])
    assert stats.cold_seconds is None
    assert stats.warm_seconds is None
