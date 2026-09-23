# Measurement methodology

## Standards we reference

Our methodology is informed by two established frameworks, adapted to our constrained scope:

- **[IETF draft-gaikwad-llm-benchmarking-methodology-01](https://datatracker.ietf.org/doc/draft-gaikwad-llm-benchmarking-methodology/)** — defines warm-up requirements (§4.5.1), streaming protocol handling (§4.6), measurement precision, and reporting formats for LLM serving systems. We adopt their warm-up and cold/warm distinction while acknowledging we don't implement their full serving stack methodology.

- **[FMwork](https://arxiv.org/abs/2508.10251)** (IBM Research) — introduces meta-metrics: quantifying the cost and accuracy of the benchmarking process itself. We adopt the principle of explicitly documenting what our measurements *don't* capture.

## What we measure

| Metric | Method | Precision |
|---|---|---|
| Response time | `time.perf_counter()` wall-clock | Milliseconds |
| Memory usage | Post-run Ollama container state | GiB (approximate) |
| Response quality | Manual evaluation | Qualitative (pass/fail/notes) |

## What we don't measure (and why)

- **Token throughput (tok/s)**: Not implemented yet — requires token counting in the response
- **Time-to-first-token (TTFT)**: Ollama's `stream: False` mode doesn't expose it
- **Peak memory**: We record post-run state, not live peak — a known limitation
- **Concurrent requests**: Single-request mode only
- **Energy consumption**: Not measured

Per FMwork's meta-metrics principle, the absence of these measurements is a documented scope decision, not an oversight.

## Historical dataset

The historical values in `data/results.csv` were extracted from local infrastructure documentation. They describe one CPU-only environment and one short French prompt:

> "Explain in five sentences the difference between Docker and a virtual machine."

They are retained as engineering observations, not presented as laboratory-grade benchmarks.

## Recommended protocol

For reproducible results:

1. **Pin versions**: Ollama version, model tag (including quantization level)
2. **Document hardware**: CPU model, RAM, context length, OS
3. **Warm-up**: Run one untimed request before measurement (per IETF §4.5.1)
4. **Consistent prompts**: Same prompt for every model comparison
5. **Repeat**: Multiple runs before drawing conclusions
6. **Separate concerns**: Cold start vs. warm latency (when implementing)
7. **Record raw data**: Preserve individual measurements alongside aggregates

## Privacy boundary

Prompts and responses must not contain personal, infrastructure-sensitive or credential-bearing data. Benchmark data should remain synthetic or public.
