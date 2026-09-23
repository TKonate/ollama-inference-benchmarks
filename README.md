# Ollama Inference Benchmarks

> **LLM benchmarks for hardware you already own — no GPU required.**

Systematic, reproducible inference benchmarks for small language models running on CPU-only machines. If you're running Ollama on a VPS, a laptop, or a used server, this project helps you measure what actually works.

## Who is this for?

- **Solo developers** choosing a model for local async tasks
- **Small teams** evaluating whether self-hosted inference beats API costs
- **Students and hobbyists** exploring local AI without buying a GPU
- **Budget-conscious ops** sizing a VPS for low-conference inference workloads

If you have a GPU, projects like [LLM-Inference-Bench](https://github.com/argonne-lcf/LLM-Inference-Bench) are better suited. This project focuses on the segment that nobody else benchmarks: constrained CPU-only environments.

## Quick start

```bash
# Run a single benchmark
python3 scripts/benchmark.py --model qwen3:1.7b --prompt "Explain Docker in five sentences."

# Compare multiple models
for model in qwen3:1.7b llama3.2:1b gemma3:1b; do
  python3 scripts/benchmark.py --model "$model" --prompt "Explain Docker in five sentences."
done
```

**Requirements:**
- Python 3.10+
- Ollama running locally or at a configured URL
- The selected model already available in Ollama

## Sample results

These benchmarks were run on a CPU-only VPS (6 GB RAM, no GPU):

| Model | Response time | RAM usage | Quality |
|---|---:|---:|---|
| Qwen3 1.7B | ~73 s | ~1.9–2.9 GiB | ✅ Best compromise |
| Llama 3.2 1B | ~57–71 s | ~1.5–4.0 GiB | ⚠️ Fast but unreliable |
| Gemma 3 1B | ~75–91 s | ~1.0–3.5 GiB | ❌ Quality doesn't justify cost |
| SmolLM2 1.7B | ~255 s | ~3.5 GiB | ❌ Too slow |

> These are observations from one constrained environment. They are not universal model benchmarks. See [Methodology](docs/methodology.md) for measurement boundaries.

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

- [ ] **Multi-backend support** — llama.cpp, vLLM (CPU mode) alongside Ollama
- [ ] **Benchmark suite** — standardized prompt set covering summarization, classification, code generation
- [ ] **CSV export** — structured results for comparison across runs
- [ ] **GitHub Actions CI** — automated linting and smoke tests
- [ ] **Docker setup** — reproducible benchmark environment
- [ ] **Cold vs. warm latency** — per IETF §4.5.1 recommendations

## Project structure

```
ollama-inference-benchmarks/
├── scripts/
│   └── benchmark.py          # Main benchmark script
├── data/
│   └── results.csv           # Recorded observations
├── docs/
│   ├── methodology.md        # Measurement protocol and boundaries
│   └── positioning.md        # Competitive landscape and niche
└── README.md
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
