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
- **规格与机制分离**：`AttentionSpec` 描述可见范围和缓存需求，Attention Backend 负责具体计算。
- **单一权威状态**：请求进度以 `num_computed_tokens` 为唯一权威，不维护容易不一致的 prefill/decode phase 状态。
- **最小代表性实现**：每类能力优先实现一个 reference backend 和一个有实际意义的 optimized backend，不建设 backend zoo。
- **可观察性内建**：每一版本都必须能够解释一次调度为何选择某些请求、分配哪些 block、发生哪些通信。

## 3. 稳定核心契约

以下契约应在 V1 冻结，并在后续版本中以扩展字段或新实现的方式演进，避免改变基本职责。

### 3.1 Request

```python
@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    num_computed_tokens: int
    status: RequestStatus
    sampling_params: SamplingParams
```

核心不变量：

```text
num_known_tokens   = len(prompt_token_ids) + len(output_token_ids)
num_pending_tokens = num_known_tokens - num_computed_tokens
0 <= num_computed_tokens <= num_known_tokens
```

### 3.2 SchedulerOutput

Scheduler 与 Executor 之间的执行计划，至少表达：

- 本轮请求集合；
- 每个请求计划计算的 token 数；
- 新请求、恢复请求和缓存命中信息；
- 每个 KV cache group 的 block table 或 block table delta；
- 本轮需要释放、分配或回滚的逻辑缓存信息。

### 3.3 ModelRunnerOutput

ModelRunner 向 Engine 返回的最小结果，至少表达：

- 每个可采样请求产生的 token；
- 可选的 logits/logprobs；
- 后续版本需要的 speculative acceptance 等执行结果；
- 执行错误和设备状态。

### 3.4 Executor

```python
class Executor(Protocol):
    def initialize(self, config) -> None: ...
    def determine_available_memory(self) -> list[int]: ...
    def initialize_kv_cache(self, config) -> None: ...
    def execute(self, plan: SchedulerOutput) -> ModelRunnerOutput: ...
    def shutdown(self) -> None: ...
```

V1 使用单进程实现；V2 增加多进程实现，但 Engine 不感知具体进程模型。

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
