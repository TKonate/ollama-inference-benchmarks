# Measurement methodology

## Historical dataset

The historical values in `data/historical-results.csv` were extracted from local infrastructure documentation. They describe one CPU-only environment and one short French prompt.

They are retained as an engineering observation, not presented as a laboratory-grade benchmark.

## Recommended future protocol

For stronger comparisons:

- pin the Ollama version;
- record the model tag and quantization;
- record CPU, RAM and context length;
- run a warm-up request before timed runs;
- repeat each prompt several times;
- measure cold and warm latency separately;
- collect peak memory rather than a single post-run snapshot;
- use a small fixed prompt suite;
- score correctness with an explicit rubric;
- preserve raw responses alongside aggregate results.

## Privacy boundary

Prompts and responses must not contain personal, infrastructure-sensitive or credential-bearing data. Benchmark data should remain synthetic or public.
