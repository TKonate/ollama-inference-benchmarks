"""Latency analysis helpers — cold vs. warm runs (IETF §4.5.1)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from benchmarks.models import BenchmarkResult


@dataclass(frozen=True)
class ColdWarmStats:
    """Aggregate latencies of a repeat-run benchmark.

    The first successful run is considered cold (model load included); the
    remaining successful runs are warm.
    """

    cold_seconds: float | None
    warm_seconds: float | None = None
    warm_runs: int = 0

    @property
    def warm_over_cold(self) -> float | None:
        """Warm latency as a fraction of cold latency (None when undefined)."""
        if self.cold_seconds and self.warm_seconds and self.cold_seconds > 0:
            return round(self.warm_seconds / self.cold_seconds, 2)
        return None


def summarize_cold_warm(results: list[BenchmarkResult]) -> ColdWarmStats:
    """Summarize repeated runs: first success is cold, the rest are warm.

    Failed runs are excluded from each bucket rather than counted as zero
    latency, so a flaky warm run does not distort the median.
    """
    if not results:
        return ColdWarmStats(cold_seconds=None)

    successful = [r for r in results if r.success]
    if not successful:
        return ColdWarmStats(cold_seconds=None)

    cold = round(successful[0].elapsed_seconds, 2)
    warm_times = [r.elapsed_seconds for r in successful[1:]]
    if not warm_times:
        return ColdWarmStats(cold_seconds=cold)

    warm_median = round(statistics.median(warm_times), 2)
    return ColdWarmStats(cold_seconds=cold, warm_seconds=warm_median, warm_runs=len(warm_times))
