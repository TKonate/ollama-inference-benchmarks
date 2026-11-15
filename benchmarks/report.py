"""Turn a campaign dataset into a published report: markdown + SVG charts.

The report is regenerated from the raw JSONL, never hand-edited. That is the
whole point of keeping raw rows: a reader can re-run ``bench report`` and get
the same tables, and a reviewer can check any number against the dataset.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.models import BenchmarkResult


@dataclass(frozen=True)
class ModelSummary:
    """Aggregated behaviour of one model across every cell of a campaign."""

    model: str
    runs: int
    failures: int
    median_ttft_ms: float | None
    median_tokens_per_second: float | None
    median_elapsed_seconds: float | None
    p95_elapsed_seconds: float | None
    median_load_seconds: float | None
    peak_rss_mib: float | None
    parameter_size: str | None
    quantization: str | None

    @property
    def failure_rate(self) -> float:
        """Fraction of runs that failed, 0.0 when nothing was attempted."""
        total = self.runs + self.failures
        return round(self.failures / total, 3) if total else 0.0


def load_dataset(path: Path) -> tuple[dict[str, Any] | None, list[BenchmarkResult]]:
    """Read a campaign JSONL file into (environment header, results)."""
    environment: dict[str, Any] | None = None
    results: list[BenchmarkResult] = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "environment":
                environment = record
            elif record.get("type") == "result":
                results.append(BenchmarkResult.model_validate(record["result"]))
    return environment, results


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return round(ordered[index], 1)


def summarize(results: list[BenchmarkResult]) -> list[ModelSummary]:
    """Aggregate results per model, warm runs only for latency statistics.

    Cold runs are excluded from the latency medians because model load is a
    one-off cost: mixing it in would make a 3-run median meaningless. Cold load
    time is reported separately.
    """
    by_model: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        by_model.setdefault(result.model, []).append(result)

    summaries: list[ModelSummary] = []
    for model, rows in sorted(by_model.items()):
        successes = [r for r in rows if r.success]
        warm = [r for r in successes if r.phase == "warm"]
        cold = [r for r in successes if r.phase == "cold"]
        latency_rows = warm or successes

        summaries.append(
            ModelSummary(
                model=model,
                runs=len(successes),
                failures=len(rows) - len(successes),
                median_ttft_ms=_median([r.ttft_ms for r in latency_rows if r.ttft_ms is not None]),
                median_tokens_per_second=_median(
                    [r.tokens_per_second for r in latency_rows if r.tokens_per_second is not None]
                ),
                median_elapsed_seconds=_median([r.elapsed_seconds for r in latency_rows]),
                p95_elapsed_seconds=_p95([r.elapsed_seconds for r in latency_rows]),
                median_load_seconds=_median(
                    [r.load_seconds for r in cold if r.load_seconds is not None]
                ),
                peak_rss_mib=max(
                    (r.server_peak_rss_mib for r in successes if r.server_peak_rss_mib is not None),
                    default=None,
                ),
                parameter_size=next(
                    (r.parameter_size for r in successes if r.parameter_size), None
                ),
                quantization=next((r.quantization for r in successes if r.quantization), None),
            )
        )
    return summaries


def _format_load(value: object) -> str:
    """Render the load-average triplet as '1.18 / 2.13 / 2.60'."""
    if isinstance(value, list | tuple) and value:
        parts = [str(item) for item in value]
        return " / ".join(parts)
    return "—"


def _fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "—"
    if unit:
        return f"{value:,.{digits}f} {unit}"
    return f"{value:,.{digits}f}"


def bar_chart_svg(
    title: str,
    values: dict[str, float],
    *,
    unit: str,
    width: int = 720,
    bar_height: int = 34,
    colour: str = "#2563eb",
) -> str:
    """Render a dependency-free horizontal bar chart as inline SVG.

    No matplotlib, no image files: the chart is text, so it diffs, renders on
    GitHub, and inherits the reader's colour scheme.
    """
    if not values:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="60"></svg>'

    label_width = 190
    padding = 16
    chart_width = width - label_width - padding * 2
    height = padding * 2 + 34 + len(values) * (bar_height + 8)
    peak = max(values.values()) or 1.0

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        f'<text x="{padding}" y="24" font-family="system-ui, sans-serif" '
        f'font-size="15" font-weight="600" fill="#111827">{_escape(title)}</text>',
    ]

    for index, (label, value) in enumerate(values.items()):
        y = 34 + padding + index * (bar_height + 8)
        bar_width = max(2.0, chart_width * (value / peak))
        parts.append(
            f'<text x="{padding}" y="{y + bar_height * 0.68}" font-family="system-ui, sans-serif" '
            f'font-size="13" fill="#374151">{_escape(label)}</text>'
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" '
            f'rx="4" fill="{colour}" opacity="0.9"/>'
        )
        parts.append(
            f'<text x="{label_width + bar_width + 8:.1f}" y="{y + bar_height * 0.68}" '
            f'font-family="system-ui, sans-serif" font-size="13" fill="#111827">'
            f"{value:,.1f} {_escape(unit)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def render_markdown(
    environment: dict[str, Any] | None,
    summaries: list[ModelSummary],
    *,
    total_runs: int,
    throughput_chart: str | None = None,
    ttft_chart: str | None = None,
) -> str:
    """Build the human-facing results page."""
    lines: list[str] = []
    lines.append("# Benchmark results")
    lines.append("")
    lines.append(
        "Generated by `bench report` from the raw dataset. Every figure below is a median "
        "of warm runs, unless stated otherwise."
    )
    lines.append("")

    if environment:
        env = environment.get("environment", {})
        options = environment.get("options", {})
        lines.append("## Environment")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|---|---|")
        lines.append(f"| CPU | {env.get('cpu_model', '—')} |")
        lines.append(f"| vCPU | {env.get('cpu_count', '—')} |")
        lines.append(
            f"| RAM | {env.get('total_ram_gib', '—')} GiB total, "
            f"{env.get('available_ram_gib', '—')} GiB free at start |"
        )
        lines.append(f"| OS / kernel | {env.get('os_name', '—')} / {env.get('kernel', '—')} |")
        lines.append(f"| Ollama | {env.get('ollama_version', '—')} |")
        lines.append(f"| Load average at start | {_format_load(env.get('load_average'))} |")
        lines.append(f"| Runs | {total_runs} |")
        if options:
            rendered = ", ".join(f"`{key}={value}`" for key, value in options.items())
            lines.append(f"| Fixed options | {rendered} |")
        lines.append("")

    lines.append("## Per-model summary")
    lines.append("")
    lines.append(
        "| Model | Params | Quant | TTFT (ms) | tok/s | Median latency (s) | p95 (s) "
        "| Cold load (s) | Peak RSS (MiB) | Failures |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        lines.append(
            f"| `{summary.model}` | {summary.parameter_size or '—'} "
            f"| {summary.quantization or '—'} "
            f"| {_fmt(summary.median_ttft_ms, '', 0)} "
            f"| {_fmt(summary.median_tokens_per_second, '', 2)} "
            f"| {_fmt(summary.median_elapsed_seconds)} "
            f"| {_fmt(summary.p95_elapsed_seconds)} "
            f"| {_fmt(summary.median_load_seconds)} "
            f"| {_fmt(summary.peak_rss_mib, '', 0)} "
            f"| {summary.failures} |"
        )
    lines.append("")

    ttft_values = {s.model: s.median_ttft_ms for s in summaries if s.median_ttft_ms is not None}
    tps_values = {
        s.model: s.median_tokens_per_second
        for s in summaries
        if s.median_tokens_per_second is not None
    }
    if tps_values:
        lines.append("## Generation speed")
        lines.append("")
        if throughput_chart:
            lines.append(f"![Median generation speed per model]({throughput_chart})")
        else:
            lines.append(bar_chart_svg("Median generation speed", tps_values, unit="tok/s"))
        lines.append("")
    if ttft_values:
        lines.append("## Responsiveness")
        lines.append("")
        if ttft_chart:
            lines.append(f"![Median time to first token per model]({ttft_chart})")
        else:
            lines.append(
                bar_chart_svg(
                    "Median time to first token", ttft_values, unit="ms", colour="#0f766e"
                )
            )
        lines.append("")

    lines.append("## What these numbers do not tell you")
    lines.append("")
    lines.append(
        "- **Quality.** Speed is not usefulness. These tables say nothing about whether a "
        "model answered correctly; read the prompt outputs in the dataset for that."
    )
    lines.append(
        "- **Concurrency.** One request at a time. Throughput under parallel load is lower "
        "and is not measured here."
    )
    lines.append(
        "- **Portability.** Figures are tied to the environment above. A 4-vCPU shared VPS "
        "is not a desktop CPU; expect large differences."
    )
    lines.append(
        "- **Noise.** A shared machine is never quiet. p95 latency is reported next to the "
        "median precisely because run-to-run spread is real."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(dataset_path: Path, out_dir: Path, *, stem: str | None = None) -> list[Path]:
    """Generate the markdown report and SVG charts, returning the written paths."""
    environment, results = load_dataset(dataset_path)
    summaries = summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or dataset_path.stem

    written: list[Path] = []
    throughput_name: str | None = None
    ttft_name: str | None = None

    tps_values = {
        s.model: s.median_tokens_per_second
        for s in summaries
        if s.median_tokens_per_second is not None
    }
    if tps_values:
        throughput_name = f"{stem}-throughput.svg"
        chart_path = out_dir / throughput_name
        chart_path.write_text(
            bar_chart_svg("Median generation speed", tps_values, unit="tok/s"),
            encoding="utf-8",
        )
        written.append(chart_path)

    ttft_values = {s.model: s.median_ttft_ms for s in summaries if s.median_ttft_ms is not None}
    if ttft_values:
        ttft_name = f"{stem}-ttft.svg"
        chart_path = out_dir / ttft_name
        chart_path.write_text(
            bar_chart_svg("Median time to first token", ttft_values, unit="ms", colour="#0f766e"),
            encoding="utf-8",
        )
        written.append(chart_path)

    markdown = render_markdown(
        environment,
        summaries,
        total_runs=len(results),
        throughput_chart=throughput_name,
        ttft_chart=ttft_name,
    )
    report_path = out_dir / f"{stem}.md"
    report_path.write_text(markdown, encoding="utf-8")
    written.insert(0, report_path)

    return written
