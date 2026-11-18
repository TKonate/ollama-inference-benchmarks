# Measurement methodology

## Standards we reference

Our methodology is informed by two established frameworks, adapted to our constrained scope:

- **[IETF draft-gaikwad-llm-benchmarking-methodology-01](https://datatracker.ietf.org/doc/draft-gaikwad-llm-benchmarking-methodology/)** — defines warm-up requirements (§4.5.1), streaming protocol handling (§4.6), measurement precision, and reporting formats for LLM serving systems. We adopt their warm-up and cold/warm distinction while acknowledging we don't implement their full serving stack methodology.

- **[FMwork](https://arxiv.org/abs/2508.10251)** (IBM Research) — introduces meta-metrics: quantifying the cost and accuracy of the benchmarking process itself. We adopt the principle of explicitly documenting what our measurements *don't* capture.

## What we measure

| Metric | Method | Precision |
|---|---|---|
| Wall-clock latency | `time.perf_counter()` around the HTTP exchange | Milliseconds |
| Time to first token (TTFT) | Streamed response; timestamp of the first non-empty chunk | Milliseconds |
| Generation speed | Ollama `eval_count / eval_duration` (§4.6 counters) | 0.01 tok/s |
| Cold model load | Ollama `load_duration`, measured after an explicit `keep_alive: 0` eviction | Microseconds, reported in seconds |
| Prompt processing | Ollama `prompt_eval_duration` | Microseconds |
| Peak server memory | RSS of the Ollama process **tree** (daemon + runner) sampled every 50 ms | MiB |
| Response quality | Manual evaluation of stored outputs | Qualitative (pass/fail/notes) |

Two details matter more than they look:

1. **The client's own RSS is meaningless.** Weights live in the backend process. An
   earlier version of this project sampled the benchmark process and reported
   numbers two orders of magnitude too small; the tracker now walks
   `/proc/<pid>/task/<pid>/children` and sums the tree, so the daemon *and* the
   runner that actually holds the weights are counted.
2. **Generation speed comes from the server, not from wall clock.** Dividing
   `completion_tokens` by total elapsed time mixes queueing and network into the
   result. `eval_duration` isolates sampling, which is the number that compares
   across models.

## What we don't measure (and why)

- **Concurrent requests**: one request at a time. Throughput under parallel load is
  a different experiment and needs a load generator.
- **Quality scores**: outputs are stored, but no automated rubric scores them. Speed
  is not usefulness, and pretending otherwise is how benchmark tables mislead.
- **Energy consumption**: no power instrumentation on the reference machines.
- **GPU behaviour**: out of scope by design — GPU benchmarking is well covered
  elsewhere (LLM-Inference-Bench).
- **Cross-machine portability**: a result belongs to the environment fingerprint
  published with its dataset. We do not claim a model is "fast" in general.

Per FMwork's meta-metrics principle, the absence of these measurements is a documented scope decision, not an oversight.

## Campaign datasets

A single run answers "what happened once". A campaign answers "what this machine
reliably does", and it is the only unit this project publishes.

```bash
bench matrix --models qwen3:0.6b,llama3.2:1b --runs 5 --prompts-per-category 2 \
             --out data/campaigns/<machine>.jsonl
bench report --dataset data/campaigns/<machine>.jsonl --out-dir docs/results
```

A campaign applies, for every run:

- **Fixed sampling options** (`temperature=0`, `seed=42`, `top_k=1`, `num_predict=256`).
  Greedy decoding makes repeats comparable and the token cap keeps one chatty model
  from dominating the wall clock. Without a cap, a "tokens per second" table compares
  verbosity instead of speed.
- **A real cold start per model**: the model is evicted with `keep_alive: 0` and the
  campaign waits for `/api/ps` to confirm eviction before the first timed run.
- **An environment fingerprint** written as the first line of the dataset: CPU model,
  vCPU count, RAM, kernel, OS, Ollama version, load average at start.
- **Raw rows, never only aggregates**: the report is regenerated with `bench report`,
  so any published figure can be traced back to the individual measurement.

Records are newline-delimited JSON: one `environment` record, one `result` record per
run, one `summary` record. The response text is stored with the timings, so quality
can be judged later without re-running anything.

## Historical dataset

The historical values in `data/results.csv` were extracted from local infrastructure documentation. They describe one CPU-only environment and one short French prompt:

> "Explain in five sentences the difference between Docker and a virtual machine."

They are retained as engineering observations, not presented as laboratory-grade
benchmarks. Campaign datasets under `data/campaigns/` supersede them: those carry
per-run rows, an environment fingerprint and machine-readable timings.

## Recommended protocol

For reproducible results:

1. **Pin versions**: Ollama version, model tag (including quantization level)
2. **Document hardware**: CPU model, RAM, context length, OS
3. **Warm-up**: Run one untimed request before measurement (per IETF §4.5.1)
4. **Consistent prompts**: Same prompt for every model comparison
5. **Repeat**: Multiple runs before drawing conclusions
6. **Separate concerns**: Cold start vs. warm latency (when implementing)
7. **Record raw data**: Preserve individual measurements alongside aggregates

## Measurement log

- **2026-11-15 — first campaign discarded.** Two harness defects were caught by
  auditing the dataset instead of trusting the summary: the `phase` field marked the
  first run of *every* prompt as cold (five of six "cold loads" per model were warm
  starts reading 0.002 s), and memory sampling followed the daemon instead of the
  detached runner (26-48 MiB reported for models occupying 300-1450 MiB). Both were
  fixed, and the campaign was re-run from scratch. The published dataset is the
  re-run; the discarded one is not kept, deliberately — a wrong number with a
  plausible shape is worse than no number.

## Privacy boundary

Prompts and responses must not contain personal, infrastructure-sensitive or credential-bearing data. Benchmark data should remain synthetic or public.
