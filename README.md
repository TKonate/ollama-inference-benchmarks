# Ollama Inference Benchmarks

Small, reproducible experiments for comparing local language models on a modest CPU-only VPS.

## Purpose

This project documents the trade-offs between response time, memory usage and answer quality when running small language models locally with Ollama.

It was created to answer a practical question:

> Which local model is useful for lightweight asynchronous tasks when compute resources are limited?

## Historical results

The following observations were collected using the same short French prompt:

> Explain in five sentences the difference between Docker and a virtual machine.

| Model | Observed response time | Observed Ollama RAM | Qualitative result |
|---|---:|---:|---|
| Qwen3 1.7B | ~1 min 13 s | ~1.9–2.9 GiB | Best overall compromise observed |
| Llama 3.2 1B | ~57 s–1 min 11 s | ~1.5–4.0 GiB | Faster, but technically unreliable answers |
| Gemma 3 1B | ~1 min 15 s–1 min 31 s | ~1.0–3.5 GiB | Quality did not justify the gain |
| SmolLM2 1.7B | ~4 min 15 s | ~3.5 GiB | Too slow and low-quality output |

These values are historical observations from one constrained environment. They are not universal model benchmarks.

## Main findings

- Fewer parameters did not automatically produce proportionally lower latency.
- CPU performance was the main bottleneck in this environment.
- Qwen3 1.7B provided the most useful quality/resource compromise among the tested small models.
- Local inference was suitable for asynchronous or repetitive tasks, but not for fluid interactive conversations.
- Memory readings from `docker stats` represented container state after runs, not scientifically measured peak consumption.

## Reproduce the experiment

The benchmark script uses only Python's standard library and the Ollama HTTP API:

```bash
python3 scripts/benchmark.py --model qwen3:1.7b --prompt "Explain Docker in five sentences."
```

Requirements:

- Python 3.10+
- Ollama running locally or at a configured URL
- the selected model already available in Ollama

The script reports elapsed time and the model response. External container metrics should be collected separately with the host's monitoring tools.

## Methodology

1. Start from a clean model state when possible.
2. Use the same prompt for every model.
3. Record elapsed wall-clock time.
4. Record qualitative issues separately from speed.
5. Treat memory measurements as observations and record how they were collected.
6. Repeat runs before drawing stronger conclusions.

## Limitations

- The historical dataset is small.
- The original tests used one short French prompt.
- Hardware, quantization, Ollama version, context length and warm-up state affect results.
- Qualitative evaluation was manual rather than scored with a formal rubric.
- Results should not be interpreted as a general leaderboard.

## Data

- [`data/results.csv`](data/results.csv) contains the documented observations.
- [`docs/methodology.md`](docs/methodology.md) explains the measurement boundaries.

## License

MIT
