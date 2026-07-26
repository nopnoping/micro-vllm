# micro-vLLM

[English](README_EN.md) | 简体中文

micro-vLLM 是一个面向学习和实验的轻量级 LLM 推理引擎。

项目保留了 vLLM V1 的核心架构设计，包括请求调度、continuous batching、
Paged KV Cache，以及 Engine、Scheduler、Executor、Worker、ModelRunner
之间清晰的职责边界；同时缩小模型和功能的覆盖范围，专注于自回归
`decoder-only`（Causal LM）模型的推理流程。

我们的目标不是重新实现一个功能完整的 vLLM，而是提供一条更短、更清晰的学习路径：
让读者能够在规模适中的代码库里理解 vLLM 的核心抽象，并在此基础上继续扩展和实验。

## 为什么做 micro-vLLM？

学习 LLM 推理系统时，通常会遇到两个不同方向的项目：

- nano-vllm 足够小，适合快速了解一次推理是如何运行起来的；但为了追求极简，
  它简化了许多系统边界，整体结构与 vLLM V1 存在较大差异。随着调度策略、
  KV Cache、模型类型和执行后端不断增加，代码较难在不重构核心流程的情况下扩展。
- vLLM 是一个面向生产环境的通用推理引擎。为了兼容大量模型架构、硬件后端、
  并行策略、量化方案和服务场景，它必须包含大量抽象、兼容逻辑和工程代码。
  这些能力对生产系统很重要，却也提高了初学者理解核心推理链路的门槛。

micro-vLLM 希望填补两者之间的空白：

> 使用接近 vLLM V1 的架构组织代码，同时只实现具有代表性的核心能力，
> 帮助读者先掌握自回归 decoder-only 模型的推理本质，再逐步理解和扩展完整的 vLLM。

## 与 nano-vllm、vLLM 的区别

| 项目 | 主要目标 | 架构与扩展性 | 功能范围 | 适合人群 |
|---|---|---|---|---|
| nano-vllm | 用尽可能少的代码实现可运行的推理引擎 | 结构极简，但与 vLLM V1 的模块边界差异较大，扩展复杂能力时容易触及核心流程 | 保留少量关键功能 | 希望快速了解最小推理实现的读者 |
| micro-vLLM | 在可读性和系统设计之间取得平衡 | 保留 vLLM V1 的核心分层，强调稳定边界和可扩展性 | 聚焦 decoder-only 推理及有代表性的调度、缓存和执行能力 | 希望系统学习 vLLM 架构并动手实验的读者 |
| vLLM | 提供高性能、生产级的通用推理服务 | 架构完整、扩展能力强，同时包含大量兼容层和工程抽象 | 覆盖多种模型、硬件、并行、量化与部署场景 | 生产用户、贡献者和需要完整能力的开发者 |

micro-vLLM 并不追求替代另外两个项目：它比 nano-vllm 更重视架构边界和后续扩展，
又通过主动限制功能范围，避免直接阅读 vLLM 时容易遇到的复杂度。

## 核心设计

一次请求的主要执行链路如下：

```text
LLM
 └── Engine
      ├── Scheduler + KVCacheManager
      └── Executor
           └── Worker
                └── ModelRunner
                     └── Model + Attention Backend
```

- **Engine**：管理请求生命周期，协调调度与执行。
- **Scheduler**：根据 token budget 选择本轮运行的请求，生成不可变的执行计划。
- **KVCacheManager**：管理逻辑 block、block table、prefix cache 以及缓存的分配与释放。
- **Executor / Worker**：隔离执行接口与设备进程的生命周期。
- **ModelRunner**：负责模型加载、设备资源、物理 KV Cache 和单轮模型执行。
- **Model / Attention Backend**：实现模型语义，并允许替换不同的 attention 算子。

这种分层延续了 vLLM V1 中控制面与执行面、逻辑缓存与物理缓存分离的设计，
使调度、缓存、模型和算子可以在相对稳定的边界上独立演进。

## 当前能力

- 同步离线批量生成；
- continuous batching 与 token-budget scheduling；
- Paged KV Cache、prefix cache 和请求抢占；
- reference、PyTorch SDPA 和 Triton paged attention 后端；
- Llama、Mistral 和 Qwen3 Causal LM 模型；
- Hugging Face 模型配置、Tokenizer 和 Safetensors 权重加载；
- 面向调度与 KV Cache 的结构化 trace；
- CPU 单元测试与 GPU/真实模型集成测试分层。

项目当前专注于 `decoder-only` 文本生成，不以兼容所有模型、硬件和生产部署场景为目标。

## 快速开始

### 安装

```bash
cd micro-vllm
python -m pip install -e .
```

运行 GPU 推理需要可用的 CUDA 环境。首次运行公开模型时，还需要能够从
Hugging Face 下载模型配置、Tokenizer 和权重。

### 运行示例

```bash
python examples/llm_generate.py
```

也可以通过公开 API 创建推理引擎：

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

## 项目原则

micro-vLLM 按以下顺序做取舍：

1. 语义正确；
2. 清晰且具有教学价值；
3. 抽象稳定、数量尽可能少；
4. 经过测量的性能优化；
5. 更广的功能覆盖。

当“支持更多功能”和“让核心推理流程保持清晰”发生冲突时，本项目优先选择后者。
