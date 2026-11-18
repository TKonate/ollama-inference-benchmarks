# Ollama Inference Benchmarks

> **LLM benchmarks for hardware you already own — no GPU required.**

**Current release: v0.5.0** — time to first token, real generation throughput, peak server memory
and reproducible campaigns. [Changelog](CHANGELOG.md).

[![CI](https://github.com/TKonate/ollama-inference-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/TKonate/ollama-inference-benchmarks/actions/workflows/ci.yml)

Systematic, reproducible inference benchmarks for small language models running on CPU-only machines. If you're running Ollama on a VPS, a laptop, or a used server, this project helps you measure what actually works.

## Who is this for?

- **Solo developers** choosing a model for local async tasks
- **Small teams** evaluating whether self-hosted inference beats API costs
- **Students and hobbyists** exploring local AI without buying a GPU
- **Budget-conscious ops** sizing a VPS for low-concurrency inference workloads

If you have a GPU, projects like [LLM-Inference-Bench](https://github.com/argonne-lcf/LLM-Inference-Bench) are better suited. This project focuses on the segment that nobody else benchmarks: constrained CPU-only environments.

## Quick start

```bash
# Install the CLI (requires Python 3.10+)
pip install -e ".[dev]"

# Or, with uv
uv sync --extra dev

# Run a single benchmark (cold latency)
bench run --model qwen3:1.7b --prompt "Explain Docker in five sentences."

# Repeat runs to measure cold vs. warm latency (first run cold, rest warm)
bench run --model qwen3:1.7b --prompt "Explain Docker in five sentences." --runs 3

# Benchmark against another backend (llama.cpp / vLLM, OpenAI-compatible API)
bench run --model qwen3:1.7b --prompt "Hello" --backend llamacpp --url http://127.0.0.1:8080

# Compare every model available on the backend with the same prompt
bench compare

# Run the standardized prompt suite (summarization, classification, code generation)
bench suite --model qwen3:1.7b

# Run a reproducible campaign: models x suite x repeats -> raw JSONL dataset
bench matrix --models qwen3:0.6b,llama3.2:1b --runs 5 --prompts-per-category 2 \
             --out data/campaigns/my-machine.jsonl

# Regenerate a published report (markdown + SVG charts) from that dataset
bench report --dataset data/campaigns/my-machine.jsonl --out-dir docs/results

# Show hardware info relevant to benchmarking
bench info
```

**Requirements:**
- Python 3.10+
- Ollama running locally or at a configured URL (`bench run --url http://host:11434`)
- The selected model already available in Ollama

### Run in Docker

```bash
docker build -t ollama-bench .

# Use the host network to reach a local Ollama instance
docker run --rm --network host ollama-bench bench run --model qwen3:1.7b --prompt "Hello"

# Run the test suite inside the container
docker run --rm ollama-bench
```

## Sample results

A full campaign — 5 models × 6 prompts × 5 repeats, greedy decoding, 256-token cap —
on a shared 4-vCPU VPS (AMD EPYC, 7.8 GiB RAM, no GPU, Ollama 0.40.16):

| Model | Params | TTFT | Generation | Median latency | Model in memory |
|---|---:|---:|---:|---:|---:|
| `qwen2.5:0.5b` | 494M | 76 ms | 25.6 tok/s | 2.2 s | 464 MiB |
| `gemma3:270m` | 268M | 157 ms | 25.3 tok/s | 4.0 s | 311 MiB |
| `qwen3:0.6b` | 752M | 91 ms | 18.1 tok/s | 2.6 s | 973 MiB |
| `smollm2:360m` | 362M | 125 ms | 12.9 tok/s | 4.4 s | 879 MiB |
| `llama3.2:1b` | 1.2B | 140 ms | 8.4 tok/s | 8.6 s | 1453 MiB |

Full tables, charts and caveats: [`docs/results/2026-11-15-vps-4vcpu.md`](docs/results/2026-11-15-vps-4vcpu.md).
Raw rows: [`data/campaigns/2026-11-15-vps-4vcpu.jsonl`](data/campaigns/2026-11-15-vps-4vcpu.jsonl).

Reading it honestly: parameter count predicts memory, not speed. The 268M model is not
faster than the 494M one, and the 1.2B model is three times slower per token than either.
Quantisation matters as much as size — `smollm2:360m` ships in F16 and lands behind models
with more parameters stored in Q4. Peak latency (p95) is 2-4× the median on this host
because the machine is shared; the report publishes both rather than hiding the spread
behind a mean.

> These numbers belong to the environment above. They are not a claim that a model is
> "fast" in general, and they say nothing about answer quality — the responses are stored
> in the dataset so it can be judged separately.

## Methodology

This project adopts a methodology-conscious approach informed by:

- **[IETF draft-gaikwad-llm-benchmarking-methodology-01](https://datatracker.ietf.org/doc/draft-gaikwad-llm-benchmarking-methodology/)** — warm-up requirements, measurement precision, reporting formats
- **[FMwork](https://arxiv.org/abs/2508.10251)** (IBM Research) — meta-metrics for quantifying what we *don't* measure

Our protocol:
1. Pin the Ollama version and model tag (including quantization)
2. Record hardware specs (CPU, RAM, context length)
3. Run a warm-up request before timed runs
4. Use the same prompt for every model
5. Record wall-clock time and memory observations
6. Label results as "historical observation" or "live run"

Full details in [`docs/methodology.md`](docs/methodology.md).

## Roadmap

- [x] **CSV export** — structured results appended to `data/results.csv` by default
- [x] **GitHub Actions CI** — automated lint, type checking (mypy strict) and tests
- [x] **Docker setup** — reproducible benchmark environment with the CLI bundled
- [x] **Benchmark suite** — standardized prompt set covering summarization, classification, code generation
- [x] **Multi-backend support** — llama.cpp and vLLM (CPU mode) via an OpenAI-compatible API alongside Ollama
- [x] **Cold vs. warm latency** — first run is cold, warm median reported (per IETF §4.5.1)
- [x] **Time to first token** — streamed responses, first chunk timestamped
- [x] **Real generation throughput** — token counts from the server's own counters, not wall clock
- [x] **Peak server memory** — RSS of the Ollama process tree sampled during the run
- [x] **Reproducible campaigns** — `bench matrix` writes raw JSONL with an environment fingerprint
- [x] **Generated reports** — `bench report` produces markdown tables and dependency-free SVG charts
- [x] **MIT license**

## Project structure

```
ollama-inference-benchmarks/
├── benchmarks/
│   ├── cli.py               # Typer CLI (run / compare / info)
│   ├── client.py            # Ollama REST API client
│   ├── metrics.py           # Response-time and memory tracking
│   └── models.py            # Pydantic request/result models
├── tests/                   # pytest suite (21 tests, CI on Python 3.10–3.12)
├── docs/
│   ├── methodology.md       # Measurement protocol and boundaries
│   └── positioning.md       # Competitive landscape and niche
├── data/
│   └── results.csv          # Recorded observations
├── .github/workflows/ci.yml # Lint & format, mypy strict, tests
├── scripts/                 # Legacy standalone benchmark script
├── Dockerfile               # Reproducible benchmark environment
└── pyproject.toml           # Packaging, `bench` entry point, ruff/mypy config
```

## Contributing

Contributions welcome. See the [roadmap](#roadmap) for priorities.

When adding benchmarks:
1. Document your hardware specs (CPU model, RAM, OS)
2. Pin the Ollama version and model tag
3. Run warm-up before timed measurements
4. Include both raw timings and qualitative observations

## License

MIT

## Background

This project was created to answer a practical question:

> Which local model is useful for lightweight asynchronous tasks when compute resources are limited?

The answer turned out to be non-obvious: fewer parameters didn't automatically produce proportionally lower latency, and CPU performance was the main bottleneck. The most useful model among those tested was Qwen3 1.7B — not because it was the fastest, but because it offered the best quality-to-resource ratio.

### Competitive context

The LLM benchmarking space is well-served for GPU-heavy environments ([LLM-Inference-Bench](https://github.com/argonne-lcf/LLM-Inference-Bench), [FMwork](https://github.com/IBM/fmwork)), and the IETF is formalizing methodology for production serving stacks. But nobody is systematically benchmarking what hobbyists and small teams actually run: quantized small models on CPU-only hardware with no budget for GPUs.

This project fills that gap. Not by competing with the big players, but by being the only project that takes constrained CPU inference seriously as a benchmarking domain.

See [`docs/positioning.md`](docs/positioning.md) for the full competitive analysis.