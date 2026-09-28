"""Typer CLI for running inference benchmarks against supported backends."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from benchmarks.analysis import summarize_cold_warm
from benchmarks.clients import get_client
from benchmarks.metrics import MemoryTracker, get_system_info
from benchmarks.models import BackendType, BenchmarkRequest, BenchmarkResult
from benchmarks.suite import CATEGORY_DESCRIPTIONS, SUITE

app = typer.Typer(
    name="bench",
    help="Run LLM inference benchmarks on CPU-only hardware.",
    no_args_is_help=True,
)
console = Console()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"


@app.command()
def run(
    model: str = typer.Option(..., help="Model tag (e.g. 'qwen3:1.7b')"),
    prompt: str = typer.Option(..., help="Benchmark prompt to send"),
    url: str = typer.Option("http://127.0.0.1:11434", help="API base URL"),
    backend: BackendType = typer.Option(
        BackendType.OLLAMA, help="Inference backend (ollama, llamacpp, vllm)"
    ),
    runs: int = typer.Option(1, min=1, help="Number of runs; the first is cold, the rest are warm"),
    timeout: int = typer.Option(600, help="Request timeout in seconds"),
    track_memory: bool = typer.Option(True, help="Track peak RSS memory during run"),
    save: bool = typer.Option(True, help="Append result to data/results.csv"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run an inference benchmark, optionally measuring cold vs. warm latency."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    client = get_client(backend, url, timeout)

    # Health check
    if not client.health_check():
        console.print(f"[red]✗ Cannot reach {backend.value} at {url}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Backend: {backend.value}[/dim]")
    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/dim]")
    console.print()

    results: list[BenchmarkResult] = []
    for run_index in range(runs):
        label = "cold" if run_index == 0 else "warm"

        # Memory tracking
        tracker = MemoryTracker() if track_memory else None
        if tracker is not None:
            tracker.start()

        request = BenchmarkRequest(
            model=model,
            prompt=prompt,
            backend=backend,
            base_url=url,
            timeout=timeout,
        )
        result = client.generate(request)

        if tracker is not None:
            mem_snapshot = tracker.stop()
            result.peak_memory_mib = mem_snapshot.rss_mib

        results.append(result)
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        console.print(f"  {status} {label:4s} run  {result.elapsed_seconds:6.2f}s")

        # Save to CSV
        if save and result.success:
            _append_csv(result, source_label=f"{label} run")

    if runs > 1:
        stats = summarize_cold_warm(results)
        cold_s = f"{stats.cold_seconds:.2f}s" if stats.cold_seconds else "—"
        warm_s = f"{stats.warm_seconds:.2f}s" if stats.warm_seconds else "—"
        console.print()
        console.print(
            f"[bold]Cold:[/bold] {cold_s}   "
            f"[bold]Warm (median of {stats.warm_runs}):[/bold] {warm_s}"
        )
        if stats.warm_over_cold and stats.warm_over_cold < 1:
            console.print(f"[dim]warm = {stats.warm_over_cold:.0%} of cold latency[/dim]")


@app.command()
def info() -> None:
    """Show system hardware info relevant to benchmarking."""
    sys_info = get_system_info()
    table = Table(title="System Info", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for key, value in sys_info.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command()
def compare(
    url: str = typer.Option("http://127.0.0.1:11434", help="API base URL"),
    backend: BackendType = typer.Option(
        BackendType.OLLAMA, help="Inference backend (ollama, llamacpp, vllm)"
    ),
    prompt: str = typer.Option("Explain Docker in five sentences.", help="Shared prompt"),
) -> None:
    """Compare all models available on the backend with the same prompt."""
    client = get_client(backend, url)
    models = client.list_models()

    if not models:
        console.print("[yellow]No models found on the backend.[/yellow]")
        raise typer.Exit()

    console.print(f"[dim]Found {len(models)} models: {', '.join(models)}[/dim]")
    console.print()

    results: list[BenchmarkResult] = []
    for model_name in models:
        console.print(f"[bold]→ Benchmarking {model_name}...[/bold]")
        tracker = MemoryTracker()
        tracker.start()
        req = BenchmarkRequest(
            model=model_name,
            prompt=prompt,
            backend=backend,
            base_url=url,
        )
        result = client.generate(req)
        mem_snapshot = tracker.stop()
        result.peak_memory_mib = mem_snapshot.rss_mib
        results.append(result)

    # Summary table
    table = Table(title="Comparison Results", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Time (s)", justify="right")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Peak RAM (MiB)", justify="right")
    table.add_column("Status")

    for r in results:
        status = "[green]✓[/green]" if r.success else "[red]✗[/red]"
        tps = f"{r.tokens_per_second}" if r.tokens_per_second else "—"
        mem_str = f"{r.peak_memory_mib}" if r.peak_memory_mib else "—"
        table.add_row(r.model, f"{r.elapsed_seconds:.1f}", tps, mem_str, status)

    console.print(table)


@app.command()
def suite(
    model: str = typer.Option(..., help="Model tag (e.g. 'qwen3:1.7b')"),
    url: str = typer.Option("http://127.0.0.1:11434", help="API base URL"),
    backend: BackendType = typer.Option(
        BackendType.OLLAMA, help="Inference backend (ollama, llamacpp, vllm)"
    ),
    timeout: int = typer.Option(600, help="Request timeout in seconds"),
    save: bool = typer.Option(True, help="Append results to data/results.csv"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run the standardized prompt suite against a single model."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    client = get_client(backend, url, timeout)

    # Health check
    if not client.health_check():
        console.print(f"[red]✗ Cannot reach {backend.value} at {url}[/red]")
        raise typer.Exit(code=1)

    total_prompts = sum(len(prompts) for prompts in SUITE.values())
    console.print(f"[bold]{model}[/bold] — {total_prompts} prompts across {len(SUITE)} categories")
    console.print()

    results: list[BenchmarkResult] = []
    for category, prompts in SUITE.items():
        console.print(f"[bold cyan]{category}[/bold cyan] — {CATEGORY_DESCRIPTIONS[category]}")
        for prompt in prompts:
            tracker = MemoryTracker()
            tracker.start()
            request = BenchmarkRequest(
                model=model,
                prompt=prompt,
                backend=backend,
                base_url=url,
                timeout=timeout,
            )
            result = client.generate(request)
            mem_snapshot = tracker.stop()
            result.peak_memory_mib = mem_snapshot.rss_mib
            results.append(result)

            status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
            tps = f"{result.tokens_per_second}" if result.tokens_per_second else "—"
            preview = prompt[:60].replace("\n", " ")
            if len(prompt) > 60:
                preview += "..."
            console.print(f"  {status} {result.elapsed_seconds:6.1f}s  {tps:>7} tok/s  {preview}")
            if save and result.success:
                _append_csv(result, silent=True, source_label="suite prompt")
        console.print()

    failures = [r for r in results if not r.success]
    if failures:
        console.print(f"[yellow]{len(failures)}/{len(results)} prompts failed.[/yellow]")
        raise typer.Exit(code=1)


def _append_csv(
    result: BenchmarkResult, *, silent: bool = False, source_label: str | None = None
) -> None:
    """Append a benchmark result to the CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0

    if source_label is None:
        source_label = f"live run {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "model",
                    "elapsed_seconds",
                    "tokens_per_second",
                    "peak_memory_mib",
                    "prompt_tokens",
                    "completion_tokens",
                    "prompt",
                    "source_type",
                ]
            )
        writer.writerow(
            [
                result.model,
                result.elapsed_seconds,
                result.tokens_per_second or "",
                result.peak_memory_mib or "",
                result.prompt_tokens or "",
                result.completion_tokens or "",
                result.prompt[:100],
                source_label,
            ]
        )

    if not silent:
        console.print(f"[dim]✓ Saved to {RESULTS_CSV}[/dim]")


if __name__ == "__main__":
    raise SystemExit(app())
