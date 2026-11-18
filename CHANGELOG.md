# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-11-15

Measured campaigns: the project stops reporting latency and starts reporting what actually happened.

### Added

- **Time to first token (TTFT)** measured on streamed responses, timestamped at the first chunk.
- **Real generation throughput**: token counts read from the server's own counters instead of a
  wall-clock estimate, which was wrong whenever the model was still loading.
- **Peak server memory**: RSS of the Ollama process tree, sampled during the run (not the client's).
- **Reproducible campaigns** (`bench matrix`): N repeats per prompt/model, raw rows in JSONL with an
  environment fingerprint, so a table can be regenerated instead of trusted.
- **Generated reports** (`bench report`): Markdown tables, p50/p95, and dependency-free SVG charts.
- **Measured dataset**: 150 runs (5 models × 6 prompts × 5 repeats) on a 4-vCPU CPU-only VPS, with
  generated report and charts under `docs/results/2026-11-15-vps-4vcpu.md`.

### Fixed

- **Cold start measured for real**: the first run of a model is now distinguished from a warm run
  instead of averaging both into one misleading number.
- **Thinking-mode runs**: a response that produced no tokens was silently counted as a success; it is
  now detected and reported as a failure.

### Tests

- 62 tests covering metrics, the matrix runner, the report generator and the client (from 43).

## Earlier releases

- `0.4.0` — multi-backend (llama.cpp, vLLM via an OpenAI-compatible API), cold/warm latency, CSV export
- `0.3.0` — standardized benchmark suite (`bench suite`, 9 prompts across 3 categories)
- `0.2.0` — README refresh, Docker image, CI, CLI restructure
- `0.1.0` — initial Typer CLI and modular package
