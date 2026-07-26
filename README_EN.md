# micro-vLLM

English | [简体中文](README.md)

micro-vLLM is a lightweight LLM inference engine built for learning and
experimentation.

It preserves the core architecture of vLLM V1, including request scheduling,
continuous batching, Paged KV Cache, and clear boundaries between the Engine,
Scheduler, Executor, Worker, and ModelRunner. At the same time, it deliberately
limits model and feature coverage to focus on autoregressive `decoder-only`
(Causal LM) inference.

The goal is not to reimplement the full vLLM feature set. Instead, micro-vLLM
provides a shorter and clearer path for understanding vLLM's core abstractions
in a reasonably sized codebase that remains suitable for extension and
experimentation.

## Why micro-vLLM?

Projects for learning about LLM inference engines often sit at one of two
extremes:

- nano-vllm is small enough to show how a minimal inference engine works.
  However, its focus on simplicity leaves out several important system
  boundaries, and its overall structure differs significantly from vLLM V1.
  Adding more scheduling policies, KV Cache strategies, model families, or
  execution backends can therefore require changes to the core workflow.
- vLLM is a general-purpose, production-grade inference engine. Supporting many
  model architectures, hardware backends, parallel strategies, quantization
  methods, and serving environments inevitably introduces substantial
  abstraction, compatibility logic, and engineering code. These capabilities
  are essential in production, but they also make the core inference path
  harder for newcomers to follow.

micro-vLLM aims to fill the gap:

> Organize the code around the architecture of vLLM V1 while implementing only
> representative core features, so readers can first understand decoder-only
> inference and then move on to extending or studying the full vLLM project.

## How does it compare?

| Project | Primary goal | Architecture and extensibility | Feature scope | Best suited for |
|---|---|---|---|---|
| nano-vllm | Build a working inference engine with as little code as possible | Very compact, but its module boundaries differ from vLLM V1 and can become difficult to extend without touching the core workflow | A small set of essential features | Readers looking for the smallest practical inference implementation |
| micro-vLLM | Balance readability with realistic system design | Preserves the core vLLM V1 layering and emphasizes stable, extensible boundaries | Representative scheduling, caching, and execution features for decoder-only inference | Readers who want to study vLLM architecture and run their own experiments |
| vLLM | Provide a high-performance, production-grade general inference engine | Comprehensive and highly extensible, with many compatibility layers and engineering abstractions | Broad model, hardware, parallelism, quantization, and deployment support | Production users, contributors, and developers who need the complete feature set |

micro-vLLM is not intended to replace either project. It places more emphasis
on architecture and extensibility than nano-vllm, while deliberately limiting
its scope to avoid the complexity that comes with studying the full vLLM
codebase.

## Architecture

The main request execution path is:

```text
LLM
 └── Engine
      ├── Scheduler + KVCacheManager
      └── Executor
           └── Worker
                └── ModelRunner
                     └── Model + Attention Backend
```

- **Engine** manages request lifecycles and coordinates scheduling with
  execution.
- **Scheduler** selects requests under a token budget and produces an immutable
  execution plan.
- **KVCacheManager** manages logical blocks, block tables, prefix caching, and
  cache allocation and release.
- **Executor / Worker** separate the execution interface from device-process
  lifecycle management.
- **ModelRunner** owns model loading, device resources, the physical KV Cache,
  and individual model runs.
- **Model / Attention Backend** implements model semantics while allowing
  different attention operators to be selected.

This structure follows the vLLM V1 separation between the control plane and
execution plane, as well as between logical and physical KV Cache management.
Scheduling, caching, models, and operators can therefore evolve independently
behind relatively stable boundaries.

## Current features

- Synchronous offline batch generation;
- continuous batching and token-budget scheduling;
- Paged KV Cache, prefix caching, and request preemption;
- reference, PyTorch SDPA, and Triton paged attention backends;
- Llama, Mistral, and Qwen3 Causal LM models;
- Hugging Face model configuration, tokenizer, and Safetensors weight loading;
- structured traces for scheduling and KV Cache behavior;
- separate CPU unit tests and GPU/real-model integration tests.

The project currently focuses on `decoder-only` text generation. Supporting
every model family, hardware platform, or production deployment scenario is
explicitly outside its scope.

## Quick start

### Installation

```bash
cd micro-vllm
python -m pip install -e .
```

GPU inference requires a working CUDA environment. The first run with a public
model also requires access to Hugging Face to download its configuration,
tokenizer, and weights.

### Run the example

```bash
python examples/llm_generate.py
```

You can also create an engine through the public API:

```python
import torch

from micro_vllm import (
    CacheConfig,
    LLM,
    LLMConfig,
    ModelConfig,
    SamplingParams,
    SchedulerConfig,
)

llm = LLM(
    LLMConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B",
            device="cuda",
            dtype=torch.float16,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=256,
            max_num_batched_tokens=64,
            max_num_seqs=2,
        ),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.8,
        ),
    )
)

try:
    outputs = llm.generate(
        ["Paged attention is"],
        SamplingParams(max_tokens=16, temperature=0.7),
    )
    print(outputs[0].output_text)
finally:
    llm.shutdown()
```

## Project principles

micro-vLLM makes trade-offs in the following order:

1. Semantic correctness;
2. Clarity and educational value;
3. Stable, minimal abstractions;
4. Measured performance improvements;
5. Broader feature coverage.

When supporting more features conflicts with keeping the core inference path
clear, this project prioritizes clarity. See
[`docs/requirements/`](docs/requirements/) for detailed design constraints and
the development roadmap.
