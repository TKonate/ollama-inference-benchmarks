# Competitive Positioning

## The landscape

LLM inference benchmarking is an active field with three established tiers:

### Academic / institutional benchmarks

| Project | Scope | Hardware focus | CPU support |
|---|---|---|---|
| [LLM-Inference-Bench](https://arxiv.org/abs/2411.00136) (Argonne, 2024) | Multi-framework, multi-hardware | A100, H100, GH200, MI300X, Gaudi2, Sambanova | No |
| [FMwork](https://github.com/IBM/fmwork) (IBM, 2025) | Systematic methodology, parameter sweep optimization | NVIDIA, AMD, Intel GPUs | No |
| [LLM-Pilot](https://research.ibm.com/publications/llm-pilot-characterize-and-optimize-performance-of-your-llm-inference-services) (IBM, 2024) | Performance prediction + hardware recommendation | Multi-GPU | No |

These projects deliver rigorous methodology but target organizations with GPU budgets. They benchmark vLLM, TensorRT-LLM, and SGLang on hardware costing $10K–$100K+.

### Standards bodies

The [IETF draft on LLM serving benchmarking](https://datatracker.ietf.org/doc/draft-gaikwad-llm-benchmarking-methodology/) (Gaikwad, 2026) defines the emerging standard for:
- Warm-up requirements and cold vs. warm measurement
- Streaming protocol handling (SSE, WebSocket, gRPC)
- Reporting formats for latency, throughput, scheduling
- Infrastructure profiles (Model Engine, Application Gateway, Disaggregated Serving)

This draft is GPU-centric by default — it defines methodology for production serving stacks, not resource-constrained environments.

### Community / practitioner benchmarks

Blog posts and Reddit threads (r/LocalLLaMA) provide scattered CPU-only measurements:
- [Gemma 4 26B CPU benchmark](https://kunalganglani.com/blog/gemma-4-cpu-inference-benchmark): 5 tok/s on a $300 Xeon
- [CPU-only LLMs: what actually works](http://insiderllm.com/guides/cpu-only-llms-what-actually-works): comparison tables across hardware tiers
- Various llama.cpp performance discussions

These are valuable but informal — no standardized methodology, no reproducible setup, no commit to consistent measurement protocols.

## The gap

Nobody is doing **systematic, reproducible, CPU-only benchmarking with documented methodology**.

Every existing project either:
1. Requires GPUs (academic/institutional)
2. Is informal and non-reproducible (blog posts)
3. Focuses on serving infrastructure, not raw inference on constrained hardware (IETF)

The hobbyist and small-team segment — people running Ollama on a VPS, a laptop, or a used server — has no dedicated benchmarking resource.

## Our niche

**"LLM benchmarks for hardware you already own — no GPU required."**

### What we are

- A reproducible benchmark suite for CPU-only inference
- Methodology-conscious (informed by IETF and FMwork, adapted to our scale)
- Focused on the 1B–8B parameter range on constrained hardware (4–32 GB RAM)
- Multi-backend: Ollama, llama.cpp, vLLM (CPU mode)

### What we are not

- A GPU benchmark suite
- A general-purpose LLM leaderboard
- A serving infrastructure evaluation

### Who this is for

| Audience | Use case |
|---|---|
| Solo developers | Choosing a model for local async tasks |
| Small teams | Evaluating whether self-hosted inference beats API costs |
| Students/hobbyists | Understanding inference trade-offs without buying hardware |
| Budget-conscious ops | Sizing a VPS for low-concurrency inference workloads |

## Methodology grounding

We don't claim to be LLM-Inference-Bench. But we take methodology seriously:

- **Warm-up**: per IETF draft §4.5.1, we recommend warm-up runs before timed measurement
- **Measurement**: wall-clock time + memory observation (we acknowledge limitations)
- **Reproducibility**: synthetic prompts, pinned Ollama version, documented hardware
- **Transparency**: we label results as "historical observation" vs "live run"

Our methodology doc explicitly references:
- IETF draft-gaikwad-llm-benchmarking-methodology-01 (warm-up, measurement precision, reporting)
- FMwork meta-metrics concept (we adopt the principle of quantifying what we *don't* measure)

## Multi-backend roadmap

Adding llama.cpp and vLLM (CPU mode) support serves two purposes:
1. **Methodology**: comparing backends on the same hardware isolates software overhead
2. **Differentiation**: no existing project benchmarks Ollama vs. llama.cpp vs. vLLM on CPU

We keep the CPU-only constraint as a scope boundary, not a limitation.

## References

1. Chitty-Venkata et al., "LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators," arXiv:2411.00136, 2024.
2. IBM Research, "FMwork: Meta-Metrics and Best Practices for System-Level LLM Inference Benchmarking," arXiv:2508.10251, 2025.
3. Gaikwad, "Benchmarking Methodology for Large Language Model Serving," draft-gaikwad-llm-benchmarking-methodology-01, IETF, August 2026.
4. Gaikwad, "Performance Benchmarking Profiles for Large Language Model Serving Systems," draft-gaikwad-llm-benchmarking-profiles-01, IETF, August 2026.
