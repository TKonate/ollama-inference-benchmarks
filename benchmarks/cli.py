"""Typer CLI for running inference benchmarks against Ollama."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from benchmarks.client import OllamaClient
from benchmarks.metrics import MemoryTracker, get_system_info
from benchmarks.models import BenchmarkRequest

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
    model: str = typer.Option(..., help="Ollama model tag (e.g. 'qwen3:1.7b')"),
    prompt: str = typer.Option(..., help="Benchmark prompt to send"),
    url: str = typer.Option("http://127.0.0.1:11434", help="Ollama API base URL"),
    timeout: int = typer.Option(600, help="Request timeout in seconds"),
    track_memory: bool = typer.Option(True, help="Track peak RSS memory during run"),
    save: bool = typer.Option(True, help="Append result to data/results.csv"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run a single inference benchmark."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    client = OllamaClient(base_url=url, timeout=timeout)

    # Health check
    if not client.health_check():
        console.print(f"[red]✗ Cannot reach Ollama at {url}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/dim]")
    console.print()

    # Memory tracking
    tracker = MemoryTracker() if track_memory else None
    if tracker:
        tracker.start()

    request = BenchmarkRequest(model=model, prompt=prompt, base_url=url, timeout=timeout)
    result = client.generate(request)

    if tracker:
        mem_snapshot = tracker.stop()
        result.peak_memory_mib = mem_snapshot.rss_mib

    # Display results
    _display_result(result)

    # Save to CSV
    if save and result.success:
        _append_csv(result)


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
    url: str = typer.Option("http://127.0.0.1:11434", help="Ollama API base URL"),
    prompt: str = typer.Option("Explain Docker in five sentences.", help="Shared prompt"),
) -> None:
    """Compare all locally available models on the same prompt."""
    client = OllamaClient(base_url=url)

    # Fetch available models
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
            data = json.load(resp)
        models = [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        console.print(f"[red]✗ Failed to list models: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not models:
        console.print("[yellow]No models found locally.[/yellow]")
        raise typer.Exit()

    console.print(f"[dim]Found {len(models)} models: {', '.join(models)}[/dim]")
    console.print()

    results = []
    for model_name in models:
        console.print(f"[bold]→ Benchmarking {model_name}...[/bold]")
        tracker = MemoryTracker()
        tracker.start()
        req = BenchmarkRequest(model=model_name, prompt=prompt, base_url=url)
        result = client.generate(req)
        mem = tracker.stop()
        result.peak_memory_mib = mem.rss_mib
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
        mem = f"{r.peak_memory_mib}" if r.peak_memory_mib else "—"
        table.add_row(r.model, f"{r.elapsed_seconds:.1f}", tps, mem, status)

    console.print(table)


def _display_result(result) -> None:
    """Pretty-print a single benchmark result."""
    if not result.success:
        console.print(f"[red]✗ Failed: {result.error}[/red]")
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan", width=18)
    table.add_column("Value", style="green")

    table.add_row("Model", result.model)
    table.add_row("Elapsed", f"{result.elapsed_seconds:.2f}s")
    if result.tokens_per_second:
        table.add_row("Tokens/s", str(result.tokens_per_second))
    if result.peak_memory_mib:
        table.add_row("Peak RAM", f"{result.peak_memory_mib} MiB")
    if result.prompt_tokens:
        table.add_row("Prompt tokens", str(result.prompt_tokens))
    if result.completion_tokens:
        table.add_row("Completion tokens", str(result.completion_tokens))

    console.print(table)
    console.print()
    if result.response_text:
        preview = result.response_text[:200]
        suffix = "..." if len(result.response_text) > 200 else ""
        console.print(f"[dim]{preview}{suffix}[/dim]")


def _append_csv(result) -> None:
    """Append a benchmark result to the CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "model",
                "elapsed_seconds",
                "tokens_per_second",
                "peak_memory_mib",
                "prompt_tokens",
                "completion_tokens",
                "prompt",
                "source_type",
            ])
        writer.writerow([
            result.model,
            result.elapsed_seconds,
            result.tokens_per_second or "",
            result.peak_memory_mib or "",
            result.prompt_tokens or "",
            result.completion_tokens or "",
            result.prompt[:100],
            f"live run {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        ])

    console.print(f"[dim]✓ Saved to {RESULTS_CSV}[/dim]")


if __name__ == "__main__":
    raise SystemExit(app())
