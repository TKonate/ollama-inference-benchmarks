"""Benchmark campaigns: models x prompt suite x repeats, written as JSONL.

A campaign is the only unit whose output is worth publishing. A single run
tells you what one prompt did once; a campaign with repeats and an environment
fingerprint tells you what the machine reliably does.

Layout of the JSONL file (one JSON object per line):

    {"type": "environment", "environment": {...}, "options": {...}, "runs": 3}
    {"type": "result", "result": {...BenchmarkResult...}}
    ...
    {"type": "summary", "results": 45, "failures": 0, "finished_at": "..."}

Keeping raw rows (not aggregates) means the report can be regenerated, and the
dataset can be re-analysed later without re-running a single inference.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.client import OllamaClient
from benchmarks.metrics import ServerMemoryTracker, find_server_pid, get_environment, runner_pids
from benchmarks.models import BenchmarkRequest
from benchmarks.suite import SUITE

logger = logging.getLogger(__name__)

# Sampling defaults applied to every campaign run. Fixed seed and greedy
# decoding make repeats comparable; the token cap bounds wall-clock time so a
# chatty model cannot dominate the campaign.
DEFAULT_OPTIONS: dict[str, object] = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 256,
    "top_k": 1,
}


@dataclass(frozen=True)
class Cell:
    """One prompt of the suite, for one model."""

    model: str
    category: str
    prompt_id: str
    prompt: str


@dataclass
class CampaignConfig:
    """Everything needed to reproduce a campaign."""

    url: str = "http://127.0.0.1:11434"
    models: list[str] = field(default_factory=list)
    runs: int = 3
    prompts_per_category: int = 1
    timeout: int = 600
    options: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_OPTIONS))
    unload_between_models: bool = True


def build_cells(config: CampaignConfig) -> list[Cell]:
    """Expand models x suite into the ordered list of cells to run."""
    cells: list[Cell] = []
    for model in config.models:
        for category, prompts in SUITE.items():
            for index, prompt in enumerate(prompts[: config.prompts_per_category]):
                cells.append(
                    Cell(
                        model=model,
                        category=category,
                        prompt_id=f"{category}-{index + 1:02d}",
                        prompt=prompt,
                    )
                )
    return cells


def _model_metadata(client: OllamaClient, model: str) -> dict[str, Any]:
    """Read model size/quantisation metadata, tolerating older servers."""
    details = client.model_details(model)
    details_block = details.get("details") or {}
    meta: dict[str, Any] = {
        "parameter_size": details_block.get("parameter_size"),
        "quantization": details_block.get("quantization_level"),
        "digest": details.get("digest"),
        "capabilities": details.get("capabilities") or [],
    }
    for running in client.running_models():
        if str(running.get("name", "")) == model or str(running.get("model", "")) == model:
            size_bytes = running.get("size")
            if isinstance(size_bytes, int):
                meta["model_size_mib"] = round(size_bytes / (1024 * 1024), 1)
            break
    return meta


def run_campaign(
    config: CampaignConfig,
    out_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every cell ``config.runs`` times and append rows to ``out_path``.

    The first run of each model is a genuine cold start: the model is evicted
    with ``keep_alive: 0`` beforehand and the campaign waits until ``/api/ps``
    confirms it is no longer resident. Every later run is warm.
    """
    emit = progress or (lambda message: None)
    client = OllamaClient(base_url=config.url, timeout=config.timeout)

    if not client.health_check():
        raise RuntimeError(f"Cannot reach an Ollama server at {config.url}")

    cells = build_cells(config)
    server_pid = find_server_pid(port=_port_of(config.url), name_hint="ollama")
    environment = get_environment(ollama_version=client.version())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_runs = len(cells) * config.runs
    completed = 0
    failures = 0
    campaign_started = time.monotonic()

    with open(out_path, "w", encoding="utf-8") as handle:
        header = {
            "type": "environment",
            "environment": environment.to_dict(),
            "options": config.options,
            "runs": config.runs,
            "server_pid": server_pid,
            "server_pid_is_daemon": True,
            "campaign_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        handle.write(json.dumps(header) + "\n")

        current_model: str | None = None
        metadata: dict[str, Any] = {}
        model_options: dict[str, object] = dict(config.options)
        cold_pending = True

        for cell in cells:
            if cell.model != current_model:
                current_model = cell.model
                cold_pending = True
                if config.unload_between_models:
                    client.unload(cell.model)
                    if not client.wait_until_unloaded(cell.model):
                        emit(
                            f"  warning: {cell.model} still resident after eviction — "
                            "the next row is not a true cold start"
                        )
                metadata = _model_metadata(client, cell.model)
                model_options = dict(config.options)
                if "thinking" in (metadata.get("capabilities") or []):
                    # Reasoning models burn the token budget before emitting a
                    # visible token, which leaves TTFT undefined and compares
                    # verbosity instead of speed. Thinking is disabled and the
                    # choice is recorded with the dataset.
                    model_options["think"] = False
                emit(f"model {cell.model} — {metadata.get('parameter_size') or '?'} ")

            for run_index in range(config.runs):
                tracker = ServerMemoryTracker(server_pid, pid_provider=runner_pids)
                tracker.start()
                request = BenchmarkRequest(
                    model=cell.model,
                    prompt=cell.prompt,
                    base_url=config.url,
                    timeout=config.timeout,
                    stream=True,
                    options=model_options,
                )
                result = client.generate(request)
                peak_rss = tracker.stop()

                result.run_index = run_index
                result.phase = "cold" if cold_pending else "warm"
                cold_pending = False
                result.category = cell.category
                result.prompt_id = cell.prompt_id
                result.server_peak_rss_mib = peak_rss
                if peak_rss is not None:
                    result.peak_memory_mib = peak_rss
                if result.success:
                    result.parameter_size = metadata.get("parameter_size")
                    result.quantization = metadata.get("quantization")
                    result.digest = metadata.get("digest")
                    if result.model_size_mib is None:
                        result.model_size_mib = metadata.get("model_size_mib")
                    if result.model_size_mib is None:
                        # The model is resident now: /api/ps can report the real footprint.
                        metadata.update(_model_metadata(client, cell.model))
                        result.model_size_mib = metadata.get("model_size_mib")
                else:
                    failures += 1

                handle.write(json.dumps({"type": "result", "result": result.model_dump()}) + "\n")
                handle.flush()
                completed += 1

                emit(
                    f"  [{completed}/{total_runs}] {cell.model} {cell.prompt_id} "
                    f"run {run_index + 1} — "
                    + (
                        f"{result.elapsed_seconds:.1f}s, ttft "
                        f"{result.ttft_ms if result.ttft_ms is not None else '—'} ms, "
                        f"{result.tokens_per_second or '—'} tok/s"
                        if result.success
                        else f"FAILED: {result.error}"
                    )
                )

        summary = {
            "type": "summary",
            "results": completed,
            "failures": failures,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - campaign_started, 1),
        }
        handle.write(json.dumps(summary) + "\n")

    emit(f"dataset written to {out_path} ({completed} runs, {failures} failures)")
    return {"results": completed, "failures": failures, "path": str(out_path)}


def _port_of(url: str) -> int:
    """Extract the port from a base URL, defaulting to Ollama's own."""
    tail = url.rsplit(":", 1)
    if len(tail) == 2:
        digits = tail[1].split("/")[0]
        if digits.isdigit():
            return int(digits)
    return 11434
