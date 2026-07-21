# micro-vLLM 版本需求总览

## 1. 项目定位

micro-vLLM 是一个用于学习和实验 LLM 推理系统的精简实现。项目以 vLLM V1 的核心思想为参考，保留 continuous batching、token-budget scheduling、Paged KV Cache、分布式推理、MoE、量化、投机解码和多模态等关键概念，但不追求生产级 vLLM 的模型、硬件和部署功能覆盖率。

项目优先级依次为：

1. 语义正确，能够通过 reference implementation 对拍。
2. 核心抽象稳定，后续版本不推翻前一版本的控制流。
3. 代码简洁、可观察、适合逐层阅读和调试。
4. 在不破坏前三项的前提下实现有代表性的 GPU 优化。

## 2. 设计原则

- **统一 token 模型**：Scheduler 不区分 prefill 队列和 decode 队列，只关心请求的已知 token、已计算 token以及本轮计划计算的 token 数。
- **控制面与执行面分离**：Scheduler 产生不可变的 `SchedulerOutput`，Executor 和 ModelRunner 只执行计划，不自行改变调度状态。
- **逻辑缓存与物理缓存分离**：Engine 侧管理 block ID、引用计数和 block table，Worker 侧持有实际 KV tensor。
- **规格与机制分离**：当前 Attention Backend 负责计算；V1.5 再引入 `AttentionSpec` 描述多 attention 类型的可见范围和缓存需求。
- **模型来源与模型语义分离**：`model_loader` 解析本地目录或 Hub、读取 config 和 checkpoint；`models` 解释架构字段、组装模型并将 tensor stream 加载到参数。
- **单一权威状态**：请求进度以 `num_computed_tokens` 为唯一权威，不维护容易不一致的 prefill/decode phase 状态。
- **最小代表性实现**：每类能力优先实现一个 reference backend 和一个有实际意义的 optimized backend，不建设 backend zoo。
- **可观察性内建**：每一版本都必须能够解释一次调度为何选择某些请求、分配哪些 block、发生哪些通信。

## 3. 稳定核心契约

以下内容记录当前实现已经形成的最小契约。后续版本优先增加新实现或在边界上增加可选
能力，不把尚未实现的生产级接口提前写入 V1 contract。

### 3.1 Request

```python
@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int]
    num_computed_tokens: int
    status: RequestStatus
    finish_reason: FinishReason | None
```

核心不变量：

```text
num_known_tokens   = len(prompt_token_ids) + len(output_token_ids)
num_pending_tokens = num_known_tokens - num_computed_tokens
0 <= num_computed_tokens <= num_known_tokens
```

### 3.2 SchedulerOutput

V1 的 Scheduler 与 Executor 之间只传递执行所需的不可变快照：

```python
@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    requires_sampling: bool
    block_table: tuple[int | None, ...]

@dataclass(frozen=True)
class SchedulerOutput:
    scheduled_requests: tuple[ScheduledRequest, ...]
    preempted_request_ids: tuple[str, ...]
```

逻辑 block 分配、释放、prefix commit 和 Request 状态仍由 Scheduler/KVCacheManager
拥有，不作为设备执行命令重复编码。多 KV cache group 在 V1.5 通过扩展执行快照表达。

### 3.3 ModelRunnerOutput

V1 的 ModelRunner 只返回按 request ID 关联的 sampled token：

```python
@dataclass(frozen=True)
class SampledToken:
    request_id: str
    token_id: int

@dataclass(frozen=True)
class ModelRunnerOutput:
    sampled_tokens: tuple[SampledToken, ...]
```

logprobs、speculative acceptance 或设备诊断只有在对应版本出现真实消费者时再扩展。

### 3.4 Executor

```python
class Executor(Protocol):
    def execute(self, plan: SchedulerOutput) -> ModelRunnerOutput: ...
```

Engine 只依赖 `execute()`。V1 的 `SingleProcessExecutor` 额外提供显式 `shutdown()`，
但生命周期方法不进入 Engine 所需的最小 protocol。V2 可以增加多进程实现和初始化
协议，而不改变 Engine 的执行调用。

### 3.5 ModelConfig 与 Model Loader

- `config.ModelConfig` 是通用、只读的 Hugging Face metadata mapping；构造时统一校验
  `architectures`，并提供通用字段读取方法，但不定义所有模型字段的并集。
- `model_loader.load_model()` 负责模型位置解析、config IO、Hub commit 固定、checkpoint
  发现和加载编排。
- `models.registry` 是 architecture name 到 model class 的静态 mapping，不提供运行时
  Registry object 或动态注册 API。
- 具体模型从 `ModelConfig` 读取自身字段，并通过 `load_weights(tensor_iterable)` 解释
  checkpoint parameter name 和参数布局。
- `model_loader` 不导入具体模型，也不解释 Llama/Qwen 的字段或参数名称。

## 4. 版本路线

| 版本 | 主题 | 新增的系统维度 | 需求文档 |
|---|---|---|---|
| V1 | 单卡推理核心 | 调度、Paged KV、执行、模型和权重 | [V1 Core Engine](./v1-core-engine.md) |
| V2 | 分布式推理 | TP、DP、PP 和多进程执行 | [V2 Distributed](./v2-distributed.md) |
| V3 | MoE | 动态路由、Expert Parallel | [V3 MoE](./v3-moe.md) |
| V4 | 量化 | 权重、激活和 KV 的数值表示 | [V4 Quantization](./v4-quantization.md) |
| V5 | 投机解码 | proposal、verification 和 rollback | [V5 Speculative Decoding](./v5-speculative-decoding.md) |
| V6 | ViT 多模态 | encoder 计算、缓存和 embedding merge | [V6 Multimodal](./v6-multimodal.md) |

## 5. 开发前置检查

- [ ] 确认 Python、PyTorch、CUDA、Triton 和 Transformers 的最低版本。
- [ ] 建立 `src/`、`tests/`、`benchmarks/`、`examples/` 和 `docs/` 目录约定。
- [ ] 配置 lint、type check、unit test 和 GPU integration test。
- [ ] 定义配置对象的分层：Model、Scheduler、Cache、Parallel、Device。
- [ ] 定义错误类型和断言策略，核心不变量失败时禁止静默恢复。
- [ ] 准备 tiny random model config，保证大部分测试不依赖下载真实权重。
- [ ] 准备与 Hugging Face Transformers 对拍的测试工具。
- [ ] 确定 benchmark 的固定输入集和结果记录格式。

## 6. 跨版本质量要求

### 6.1 正确性

- [ ] 所有 optimized backend 都必须能与 reference backend 对拍。
- [ ] greedy decoding 能与 Transformers 在容许误差范围内逐 token 对齐。
- [ ] 每项优化均提供关闭开关，关闭后仍可运行正确性基线。
- [ ] Scheduler、BlockPool、BlockTable 使用 property/invariant tests。
- [ ] 分布式、量化、投机和多模态功能都必须提供最小端到端测试。

### 6.2 可观察性

- [ ] 每个 engine step 可输出结构化 trace。
- [ ] trace 至少包含 token budget、request allocation、block 变化和终止原因。
- [ ] 分布式版本记录 collective 类型、参与 group 和传输 tensor shape。
- [ ] 支持关闭详细 trace，关闭时不得明显影响热路径性能。

### 6.3 性能指标

- [ ] 记录 request throughput 和 output token throughput。
- [ ] 记录 TTFT、ITL 和端到端 latency。
- [ ] 记录 GPU peak memory 和 KV block utilization。
- [ ] 记录 prefix cache hit、preemption 和 recompute token 数。
- [ ] 每个版本保留前一版本的 benchmark 基线，避免无解释的严重退化。

### 6.4 文档

- [ ] 每个核心对象说明职责、所有权、线程/进程边界和不变量。
- [ ] 每个版本提供一个最小可运行示例。
- [ ] 复杂算法提供面向学习者的逐步说明和可视化 trace 示例。
- [ ] 明确记录简化点及其与生产级 vLLM 的差异。

## 7. 全局非目标

以下内容默认不属于 V1–V6 的交付范围，除非后续单独立项：

- Ray executor 和多套分布式运行时；
- OpenAI 兼容 API 的完整实现；
- CPU KV swap、KV offload 和 disaggregated serving；
- KV connector、远程 KV 传输；
- LoRA、prompt adapter；
- beam search、best-of、structured/guided output；
- context parallel、decode context parallel；
- Mamba、SSM、encoder-decoder 模型；
- 所有模型、量化格式和硬件平台的兼容性；
- production-grade fault tolerance、elastic scaling 和安全隔离。

## 8. 完成状态约定

- `[ ]`：尚未实现。
- `[x]`：代码、测试和必要文档均已完成。
- 功能只有在其验收项全部通过后才可标记完成。
- P0 功能全部完成后版本才可发布；P1 功能允许在对应小版本补充；P2 只作为未来扩展记录，不阻塞版本发布。
