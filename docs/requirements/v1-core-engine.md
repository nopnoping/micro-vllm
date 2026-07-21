# V1：单卡 Core Engine 需求清单

## 1. 版本目标

在单 GPU 上建立完整、可验证的 LLM serving 核心，形成后续分布式、MoE、量化、投机解码和多模态版本共同依赖的稳定控制流与数据面。

V1 必须回答以下问题：

- 请求如何从 waiting 进入 running 并最终结束；
- 如何按照 token budget 将 prefill 和 decode 放入同一个 batch；
- KV cache 如何按 block 分配、共享、释放和抢占；
- Scheduler 如何生成与设备执行解耦的执行计划；
- ModelRunner 如何把计划转换为设备输入并访问 physical KV cache；
- 通用模型配置、架构实现和 Model Loader 如何分工并加载 Hugging Face 权重。

## 2. 当前实现顺序

| 里程碑 | 内容 | 状态 |
|---|---|---|
| V1.1 | Request、统一 Scheduler、Engine control loop、Executor contract | 已完成 |
| V1.2 | 单类型 full-attention 逻辑 Paged KV Cache、prefix cache、preemption/recompute | 主体已完成 |
| V1.3 | Physical KV Cache、reference full attention、Tiny Llama execution、ModelRunner、Worker、SingleProcessExecutor | 已完成 |
| V1.4 | Model Loader、真实 Llama/Qwen checkpoint、完整 sampling 和离线 API | 当前阶段 |
| V1.5 | sliding window、chunked-local、多 KV cache group、Mistral/hybrid model | 延后实现 |
| V1.6 | optimized attention、CUDA Graph/compile、benchmark 和性能收尾 | 待实现 |

## 3. V1.1：Request 与 Engine Control Loop

### Request 生命周期

- [x] Request 包含稳定 ID、prompt tokens、output tokens、sampling params 和状态。
- [x] 请求状态包含 `WAITING`、`RUNNING`、`FINISHED`；抢占后回到 `WAITING`。
- [x] `num_computed_tokens` 是请求计算进度的唯一权威字段。
- [x] prompt/output/known/pending token 数由 Request 统一计算。
- [x] Request 拥有输入 token list 的副本，外部修改不能绕过生命周期操作。
- [x] 空 request ID、空 prompt 和负 token ID 被明确拒绝。
- [x] 支持 EOS、LENGTH 和 ABORTED finish reason。
- [x] preempt 只重置 computed progress，保留 prompt/output tokens。
- [x] resume 可以从零或 prefix-cache hit boundary 恢复。
- [x] 支持 max model length 的 admission、调度和终止语义。

### Engine

- [x] Engine 提供逐 step 驱动接口，允许上层实现 streaming。
- [x] 支持 Engine 运行过程中动态添加 tokenized request。
- [x] Engine 严格执行 `schedule -> execute -> update`。
- [x] 空 execution plan 不调用 Executor。
- [x] 提供 `has_unfinished_requests()`。
- [x] 支持 abort waiting/running request。
- [x] Engine 不直接访问 CUDA device 或 model module。
- [ ] `LLM` 提供 tokenizer-backed 离线批量 `generate()`。
- [ ] 提供异步迭代器形式的 streaming API。
- [ ] 支持累计与增量两种文本输出形式。

### 执行契约

- [x] 定义稳定的最小 Executor protocol。
- [x] `SchedulerOutput` 和嵌套 ScheduledRequest 使用不可变数据结构。
- [x] SchedulerOutput 不包含 Request 或设备 tensor。
- [x] sampled token 通过 request ID 显式关联，不依赖隐含 batch index。
- [x] executor output 校验 unknown、duplicate 和不需要采样的 request。
- [x] invalid executor result 不会部分提交 Request 状态。
- [x] 同一 plan 不能重复 update，equal-but-foreign plan 被拒绝。
- [x] 实现真实 `SingleProcessExecutor`。

## 4. V1.1：统一 Token-Budget Scheduler

### 调度语义

- [x] Scheduler 同时维护 waiting 和 running 请求，但不维护独立 prefill/decode 状态机。
- [x] 每一步以 `max_num_batched_tokens` 作为全局 token budget。
- [x] 每一步以 `max_num_seqs` 作为请求数上限。
- [x] Scheduler 为每个请求输出 `num_scheduled_tokens`。
- [x] 每请求计划 token 数不超过其 pending token 数。
- [x] running 请求优先于新 waiting 请求。
- [x] waiting request 按 FIFO admission。
- [x] running collection 保持稳定顺序。
- [x] 长 prompt 可以跨多个 step 执行 chunked prefill。
- [x] budget 不足时截断 prefill chunk，而不是阻塞整个 request。
- [x] 同一个 plan 可以混合 decode token 和 prefill chunk。
- [x] prompt 中间 chunk 不产生 sampled token。
- [x] 只有计算追上最后一个已知 token 时才要求采样。
- [x] pending plan 重试不会重复分配 cache block。
- [x] 调度同时遵守 max model length。

### 策略与可扩展性

- [x] KV 资源不足时确定性抢占最新、尚未加入当前 plan 的 running victim。
- [x] 无法容纳单 request 时明确失败，不产生无限 empty-schedule loop。
- [ ] 将 victim selection 从 Scheduler 主循环提取为独立 policy。
- [ ] 记录未被选择 request 的调度原因。
- [ ] 支持配置单 request 最大 prefill chunk。
- [ ] 支持简单 request priority。

## 5. V1.2：单类型逻辑 Paged KV Cache

本阶段只支持一个 full-attention cache group。Engine 侧管理 logical/physical block ID，不持有真实 K/V tensor。

### CacheConfig

- [x] 配置固定 block size 和 physical block 数。
- [x] 配置是否启用 prefix cache。
- [x] prefix cache 关闭时不进行 lookup/commit。
- [x] 非正 block size/count 被拒绝。
- [x] CacheConfig 不包含 dtype、head shape 或 CUDA object。

### BlockPool

- [x] 管理固定范围的 physical block ID。
- [x] 使用确定性的 FIFO free queue。
- [x] 每个 physical block 维护 refcount 和可选 block hash。
- [x] refcount 为零的 block 进入 free queue并可被重新分配。
- [x] cached zero-reference block 在 physical storage 复用前仍可命中。
- [x] physical block 被复用时清理旧 hash identity。
- [x] cached block acquire 增加 request reference。
- [x] 批量新分配和 cached block acquire 是原子的。
- [x] allocation capacity failure 不改变 free queue、refcount 或 block table。
- [x] free block snapshot 为不可变、确定性顺序。
- [x] BlockPool 不持有 Request 或实际 KV tensor。
- [ ] 显式检测 invalid block ID、double free、负 refcount 和重复 free-queue entry。

### BlockTable

- [x] KVCacheManager 为每个 request 维护独立 logical-to-physical block table。
- [x] 支持追加、替换、查询 mapping 和完整释放。
- [x] 支持 `None` 表示 unmapped logical block。
- [x] snapshot 不暴露可修改内部 list。
- [x] BlockTable 不直接修改 BlockPool refcount。
- [x] 在 ModelRunner 中建立唯一的 token position → logical block → physical slot helper。

### Prefix Cache

- [x] 使用确定性 BLAKE2 hash，不使用进程随机化的 Python `hash()`。
- [x] hash 包含 parent block hash，完整表达 prefix chain。
- [x] hash 包含 token block、block size、cache namespace 和 cache group ID。
- [x] 只有成功计算并提交的完整 block 进入 cache。
- [x] partial block 不共享，不实现 copy-on-write。
- [x] lookup 从第一个 block 连续命中，遇到 miss 后停止。
- [x] prefix discovery 与 request ownership acquire 分离。
- [x] request acquire cached block 时增加 refcount。
- [x] 相同 hash 的 duplicate computed block 被 canonical block 替换并释放。
- [x] prompt 恰好按 block 对齐时保留最后一个 block 重新计算，以获得 logits。
- [x] request 释放后 zero-reference cached block 保持可回收，并在复用时失效。
- [x] cache namespace/group 不兼容时不会误共享。

### KVCacheManager

- [x] 为 request 注册和删除 BlockTable。
- [x] 提供 prefix lookup。
- [x] 提供按 computed + scheduled token 数计算缺失 block 数。
- [x] 提供原子 `allocate_slots()`。
- [x] partial block 有剩余容量时不新增 physical block。
- [x] 跨 block boundary 时只分配缺失 block。
- [x] allocation result 包含不可变 block table snapshot 和 new block IDs。
- [x] 提供 `free_request_blocks()`，用于保留 request table 后 recompute。
- [x] 提供 `free_request()`，释放引用并删除 request table。
- [x] free 一个 request 不影响其他 request 或共享 block 的其他引用。
- [x] repeated recompute cycle 不重复释放 block reference。
- [x] unknown/repeated free 行为明确，不静默 double release。
- [x] 不实现 CPU swap/offload。

### Scheduler 集成和抢占

- [x] Scheduler 显式依赖独立 KVCacheManager，不使用隐藏的无限 cache fallback。
- [x] waiting request 只有 slot allocation 成功才进入 RUNNING。
- [x] ScheduledRequest 携带只读 physical block table snapshot。
- [x] sampled token 在下一 step 才获得 KV slot。
- [x] executor validation 失败时保留 plan reservation并允许重试。
- [x] successful update 后 commit 完整 prefix blocks。
- [x] request finish 后释放所有 request-owned KV references。
- [x] abort waiting/running/pending request 安全释放其 cache state。
- [x] abort 一个 pending request 不释放其他 plan request 的 reservation。
- [x] cache capacity 不足时抢占并释放 victim blocks。
- [x] victim 回到 WAITING 并从零 recompute。
- [x] victim resume 时可以重新命中 prefix cache。
- [x] 抢占过程不 double free，abort 被抢占 request 不泄漏 block。
- [x] 使用随机初始化 tiny Llama 执行链路验证 prefix cache on/off 输出一致。
- [x] 使用随机初始化 tiny Llama 执行链路验证 preempt/recompute 与无抢占输出一致。

## 6. V1.3：Full-Attention Model Execution（已完成）

### Physical KV Cache

- [x] ModelRunner 为每个 decoder layer 创建 physical KV cache tensor。
- [x] KV tensor 的 block dimension 与 CacheConfig physical block ID 一一对应。
- [x] tensor shape 表达 K/V、block、block offset、KV heads 和 head dim。
- [x] physical cache dtype/device 由 ModelRunner execution config 决定。
- [x] ModelRunner 拥有 tensor，Scheduler/KVCacheManager 不访问 tensor。
- [x] block 被逻辑释放后不强制清零，slot overwrite 和 block table 保证隔离。

### ModelInput 与 AttentionMetadata

- [x] 将 SchedulerOutput 转换为 flattened input IDs。
- [x] 生成每个 token 的 absolute position。
- [x] 使用唯一 helper 生成 slot mapping。
- [x] 将不同长度 block tables pad 为只读 metadata tensor。
- [x] 生成 query start locations 和每 request context length。
- [x] 生成 sampling row indices。
- [x] 支持 request 具有不同 scheduled token 数的 mixed batch。
- [x] 不建立 prefill/decode 两套 ModelInput。

### Reference Full Paged Attention

- [x] 将当前 query token 的 K/V 写入 slot mapping 指定位置。
- [x] 根据 request block table 读取完整 causal history。
- [x] chunked prefill 同时读取历史 KV 和当前 chunk。
- [x] decode 读取完整 full-attention context。
- [x] 支持 MHA、GQA 和 MQA。
- [x] softmax 使用数值稳定 dtype。
- [x] reference backend 不访问 Scheduler、Request 或 KVCacheManager。
- [x] full prefill、chunked prefill 和 decode 与 dense causal reference 对拍。

### Tiny Llama

- [x] 使用通用只读 `ModelConfig` 保存 HF metadata，由 Llama 读取自身所需字段。
- [x] `ModelConfig` 不定义所有模型专属字段的并集。
- [x] 实现 RMSNorm。
- [x] 实现标准 Llama RoPE。
- [x] 实现单卡 dense Linear、Embedding 和 LM head。
- [x] 实现 SwiGLU MLP。
- [x] 实现 Q/K/V/O projection 和 full attention layer。
- [x] 实现 decoder layer residual/pre-norm 结构。
- [x] 实现多层 Llama model 和 final RMSNorm。
- [x] 模型 `forward()` 与 `compute_logits()` 分离。
- [x] 与 Transformers tiny Llama 的 layer、hidden state 和 logits 对拍。

### Sampler、ModelRunner、Worker 和 Executor

- [x] 实现 greedy sampler。
- [x] sampler 输出通过 request ID 显式关联。
- [x] ModelRunner 构造统一 flattened batch 和 attention metadata。
- [x] ModelRunner 只对 sampling rows 计算/收集 logits。
- [x] ModelRunner 不修改 Request、SchedulerOutput 或 Scheduler 状态。
- [x] ModelRunner 拥有模型初始化、placement、warmup/profile、physical KV cache
      allocation、执行和 device tensor 生命周期。
- [x] Worker 只编排 device process 和 ModelRunner 生命周期。
- [x] Worker 不直接构造、加载或持有 model。
- [x] ModelRunner 根据 CUDA warmup peak 和 GPU memory utilization 推导实际 block 数。
- [x] ModelRunner 返回实际 `CacheConfig`，control plane 再据此创建逻辑 block pool。
- [x] Worker 提供显式 shutdown。
- [x] 实现 SingleProcessExecutor 并只委托 Worker。
- [x] Engine 使用真实 SingleProcessExecutor 完成 tiny-model generation。

### 执行正确性

- [x] 一次 full prefill 与多次 chunked prefill 的最后 logits/token 一致。
- [x] prefill + decode 与完整序列 forward 一致。
- [x] mixed decode/prefill batch 与逐 request 独立执行一致。
- [x] prefix cache on/off 输出一致。
- [x] preempt/recompute 与无抢占输出一致。
- [x] physical block 释放和复用后不存在 stale KV 污染。
- [x] CPU FP32 reference tests 通过。
- [x] 单 GPU eager tests 通过。

## 7. V1.4：权重加载、模型覆盖与完整 Sampling（当前阶段）

| 子阶段 | 内容 | 状态 | 技术清单 |
|---|---|---|---|
| V1.4A | Safetensors Checkpoint Reader | 已完成 | [清单](../implementation/v1-checkpoint-source-todo.md) |
| V1.4B | Model Loader 与 HF checkpoint | 已完成 | [清单](../implementation/v1-model-loader-todo.md) |
| V1.4C | 真实 Llama checkpoint 对拍与 Engine 集成 | 当前阶段 | [清单](../implementation/v1-real-llama-integration-todo.md) |
| V1.4D | Qwen dense model 与 checkpoint | 待实现 | 待生成 |
| V1.4E | 完整 Sampling 与 tokenizer-backed `LLM.generate()` | 待实现 | 待生成 |

V1.4B 的代码、单文件/分片/tied checkpoint 测试和 CPU/GPU 数值对拍已经在远端完成
验证。V1.4C Phase 4 已将本地 HF-format tiny Llama checkpoint 接入 ModelRunner、
SingleProcessExecutor 和 Engine；公开固定 commit checkpoint 验收留在 Phase 5。

### 当前模型加载架构

```text
model name/path
    -> ModelInitializer
    -> ModelRunner.initialize_model()
    -> model_loader.load_model()
        -> local directory or pinned Hugging Face commit
        -> config.json -> read-only ModelConfig
        -> models.registry -> model class
        -> SafetensorsCheckpoint.iter_tensors()
        -> model.load_weights(tensor iterable)
    -> ModelRunner warmup/profile
    -> available GPU KV bytes / bytes per block -> actual num_blocks
    -> ModelRunner allocates and owns physical KV tensor storage
    -> Worker exposes actual CacheConfig to the control plane
```

- `config` 拥有共享、只读的 `ModelConfig`；构造时统一校验 `architectures` 并提供
  通用字段读取方法，不解释 Llama/Qwen 专属语义。
- `model_loader` 拥有路径/Hub 解析、config IO、commit 固定、checkpoint 下载与遍历，
  但不导入具体模型或解释参数布局。
- `models.registry` 是不可变的 architecture → model class mapping，不建立 Registry
  object、动态 `register()` 或 plugin discovery。
- 具体模型负责从 `ModelConfig` 读取字段、组装网络，并在 `load_weights()` 中解释
  checkpoint parameter names。
- checkpoint reader 一次只打开一个 safetensors shard；模型收到 tensor 后直接复制到
  parameter，不维护独立的全量 missing/unexpected/duplicate 状态机。
- Worker 接收延迟 `ModelInitializer`，但只负责将其交给 ModelRunner；ModelRunner
  调用 initializer、持有模型，并负责 device/dtype placement 和 eval mode。
- ModelRunner 在最终 KV cache 分配前执行 warmup；CUDA 路径测量 peak activation，
  按 GPU memory utilization 扣除 non-KV memory，再按模型 KV block bytes 推导容量。
- `initialize_kv_cache()` 不接收外部分配的 cache 或最终 block 数；它分配并持有
  physical KV tensor，返回实际 `CacheConfig` 给 control plane 建立同容量逻辑 block pool。
- CPU reference path 无 GPU memory 可 profile，使用 `max_num_blocks` 作为确定性容量；
  CUDA 路径只将其作为可选上限，实际容量仍来自 profile。

### 相对初始清单的设计调整

| 初始设想 | 当前实现 | 调整原因 |
|---|---|---|
| 每个模型先转换为专属 config dataclass | 通用只读 `ModelConfig` 保存原始 metadata，具体模型读取自身字段 | 避免建立跨模型字段并集和重复 config adapter |
| Registry object + resolved model factory | 静态不可变 architecture → model class mapping | 当前只有静态内置模型，不提前建设动态注册框架 |
| `weights` package 和独立 parameter loader | `model_loader` 负责来源/IO/编排，具体模型实现 `load_weights()` | 参数名称和布局属于模型/layer 语义 |
| Loader 汇总 missing/unexpected/duplicate/shape 状态 | checkpoint 结构由 reader 校验；模型按名称直接查找并用 `copy_` 复制 | 首版保留 Python/PyTorch 原生错误，减少重复状态机 |
| V1 Loader 接收 size=1 `ParallelContext` | V1 不引入 parallel context | shard/placement 语义推迟到 V2 的真实 TP/PP 边界 |
| checkpoint source 只面向本地目录 | 顶层 Model Loader 同时支持本地目录和 commit-pinned Hub repo | source reader 仍保持本地；Hub 只负责把远端文件解析到本地 cache |

这些调整不改变 Scheduler、Paged KV Cache、ModelRunner 或 Attention 的 V1 核心控制流。
后续分布式、量化和多模态继续沿用“Loader 提供 tensor stream，模型/layer 解释布局”的
方向扩展。

### Model Loader 和模型接口

- [x] 通用 `ModelConfig` 保留只读 HF metadata，不定义模型专属字段全集。
- [x] `models` registry 根据 Hugging Face `architectures` 选择实现。
- [x] registry 使用静态不可变 mapping，不提供动态注册机制。
- [x] 模型类负责架构差异，不把模型判断放入 Scheduler/ModelRunner。
- [x] 模型 `forward()` 与 `compute_logits()` 职责分离。
- [x] 模型计算与 checkpoint IO 职责分离。
- [x] 模型通过 `load_weights(tensor iterable)` 拥有参数名称和布局语义。
- [x] Model Loader 支持本地目录、HF repo id 和可选静态 alias mapping；当前没有内置 alias。
- [x] 公共 RMSNorm、RoPE、MLP、Embedding、Linear 和 Attention layer 可复用。
- [x] 不实现难以理解的万能声明式 Transformer builder。

### 权重加载

- [x] 支持单文件 safetensors。
- [x] 支持 safetensors index 和多 shard 流式加载。
- [x] 避免同时将全部 checkpoint tensor 常驻 CPU 内存。
- [x] 支持本地目录和 Hugging Face Hub 延迟 checkpoint 下载。
- [x] config 和 checkpoint 固定到同一个 Hub commit。
- [x] 支持 tied embedding/lm_head。
- [x] 当前分离的 Q/K/V 和 gate/up checkpoint 参数直接映射。
- [x] 参数名称映射后直接复制，不维护额外的完整性校验状态。

### 模型覆盖

- [ ] 加载真实 Llama-family dense checkpoint。
- [ ] 加载真实 Qwen2/Qwen3 dense checkpoint。
- [ ] Llama 和 Qwen greedy 输出与 Transformers 逐 token 对拍。
- [ ] 至少两个真实小型模型族完成端到端生成。
- [x] 本地标准 HF-format GQA Llama checkpoint 经 ModelRunner initializer 加载并完成
      CPU/GPU Engine generation。
- [x] loaded Llama 的 full/chunked prefill 和多 token greedy decode 与 Transformers
      逐 token 一致。
- [x] loaded Llama 的 mixed batch、prefix cache、preemption/recompute、block reuse 和
      finish/abort KV 生命周期通过端到端测试。

### Sampling 和离线 API

- [x] greedy decoding。
- [ ] temperature sampling。
- [ ] top-k sampling。
- [ ] top-p sampling。
- [ ] per-request random generator/seed。
- [x] EOS、ignore EOS 和 max tokens 控制流。
- [x] max model length admission 和终止原因。
- [ ] tokenizer-backed `LLM.generate()`。

### 明确不实现的 sampling 功能

- [x] 不实现 beam search。
- [x] 不实现 best-of 和 parallel sampling。
- [x] 不实现 structured/guided decoding。
- [x] 不实现用户自定义 logits processor。

## 8. V1.5：多 Attention 类型与多 KV Cache Group（延后）

### AttentionSpec

- [ ] 定义不可变 AttentionSpec contract。
- [ ] 实现 `FullAttentionSpec`。
- [ ] 实现 `SlidingWindowSpec(window_size)`。
- [ ] 实现 `ChunkedLocalSpec(chunk_size)`。
- [ ] 明确定义每种 spec 的 query-key 可见范围。
- [ ] 每个 attention layer 声明自己的 spec，不由 Scheduler 推断模型类型。
- [ ] cache manager 与 attention mask 使用同一份 spec 参数。

### 多 Cache Group

- [ ] 相同 spec 的 layer 组成一个 KVCacheGroup。
- [ ] 每个 request 为每个 cache group 拥有独立 block table。
- [ ] SchedulerOutput 能表达多个 group 的 block table。
- [ ] full group 保留完整历史。
- [ ] sliding/local group 可以回收不可见的旧 block。
- [ ] 不同 group 的 physical cache layout 与 block ID 生命周期明确。

### Reference correctness

- [ ] sliding window attention 与显式 dense mask 对拍。
- [ ] chunked-local attention 与显式 dense mask 对拍。
- [ ] full + local hybrid tiny model 正确执行。
- [ ] 使用 Mistral 或等价模型验证 sliding window。
- [ ] 使用 synthetic tiny model 验证 full + chunked-local hybrid。
- [ ] 不同 block size、window/chunk size 下结果一致。

## 9. V1.6：Optimized Backend 与性能收尾

### Optimized Attention

- [ ] 至少提供一个 GPU prefill backend。
- [ ] 至少提供一个 paged decode attention backend。
- [ ] optimized chunked prefill 能读取历史 paged KV 和当前 chunk。
- [ ] optimized backend 与 reference backend 在规定容差内对拍。
- [ ] backend 可独立关闭并回退到 reference path。
- [ ] 优化项不得进入 Scheduler 核心抽象。

### P1 优化

- [ ] 增加 CUDA Graph 或 `torch.compile` 实验开关。
- [ ] 支持配置单 request 最大 prefill chunk。
- [ ] 调度/metadata 构造不成为明显 CPU 瓶颈。

## 10. 可观察性与测试

### 已有测试覆盖

- [x] Scheduler 永不超过 token 和 sequence budget。
- [x] 长 prompt 被切成多个 chunk。
- [x] decode 和 prefill 可以进入同一个 execution plan。
- [x] sampled token 只在下一 step 获得 cache slot。
- [x] BlockPool 使用固定随机序列进行生命周期 invariant test。
- [x] KV allocation capacity failure 保持原子性。
- [x] prefix hash、lookup、commit、canonical reuse 和 namespace isolation 测试。
- [x] prefix hit 后 admission 和 block-aligned prompt 重算测试。
- [x] preemption victim selection、recompute、prefix resume 和 abort 测试。
- [x] finish/abort/pending-plan 的 block release 测试。
- [x] mixed batch 与逐 request 的随机 tiny Llama 执行结果一致。
- [x] prefix cache on/off 的随机 tiny Llama 输出一致。
- [x] preempt/recompute 与无抢占的随机 tiny Llama 输出一致。
- [x] full attention 与 dense causal reference 对拍。
- [x] tiny Llama hidden state、logits 和 GPU FP16 与 Transformers 对拍。

### 待补测试

- [ ] sliding/chunked-local attention 与 dense reference 对拍。
- [ ] 固定 commit 的公开 Llama checkpoint 端到端 GPU 验收。
- [ ] Qwen 和 Mistral 外部 checkpoint 端到端加载测试。

### Trace 和指标

- [ ] 每一步记录 token budget 和已使用 budget。
- [ ] 记录每 request scheduled token 数和未调度原因。
- [ ] 记录 admission、preemption、resume 和 finish。
- [ ] 记录 block allocate/free/share 和 refcount 变化。
- [ ] 记录 prefix hit block/token 数。
- [ ] 记录 mixed batch 中 prefill/decode token 构成。
- [ ] 记录 physical KV cache shape、dtype、字节数和 utilization。
- [ ] 记录 request throughput、output token throughput、TTFT 和 ITL。

## 11. V1 明确不包含

- 多 GPU 和多进程 Worker；
- TP、PP、DP、EP；
- CPU KV swap/offload；
- disaggregated serving 和 KV connector；
- MoE；
- 权重或 KV 量化；
- speculative decoding；
- multimodal input；
- LoRA；
- OpenAI API server；
- Ray、RPC framework、故障恢复；
- 多套 optimized attention backend。

## 12. V1 发布验收

- [ ] 单 GPU 可加载至少两个真实的小型 dense 模型族并生成文本。
- [x] decode 与 chunked prefill 在同一个随机 tiny Llama batch 中正确执行。
- [x] 任意 Scheduler step 严格遵守 token 和 sequence budget。
- [x] KV 空间不足时能抢占并最终正确完成随机 tiny Llama 请求，无 block 泄漏。
- [x] prefix cache on/off 的随机 tiny Llama 输出一致。
- [x] full attention 有独立 reference correctness test。
- [ ] sliding window 和 chunked-local 有独立 reference correctness test。
- [ ] reference 与 optimized attention 通过数值对拍。
- [ ] 核心对象职责、不变量和一次完整 request trace 已文档化。
- [ ] benchmark 记录 throughput、TTFT、ITL、显存和 KV utilization。
