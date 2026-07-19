# V1：单卡 Core Engine 需求清单

## 1. 版本目标

在单 GPU 上建立完整、可验证的 LLM serving 核心，形成后续分布式、MoE、量化、投机解码和多模态版本共同依赖的稳定控制流与数据面。

V1 必须回答以下问题：

- 请求如何从 waiting 进入 running 并最终结束；
- 如何按照 token budget 将 prefill 和 decode 放入同一个 batch；
- KV cache 如何按 block 分配、共享、释放和抢占；
- Scheduler 如何生成与设备执行解耦的执行计划；
- ModelRunner 如何把计划转换为设备输入并访问 physical KV cache；
- 模型如何声明 attention/KV 需求并加载 Hugging Face 权重。

## 2. 当前实现顺序

| 里程碑 | 内容 | 状态 |
|---|---|---|
| V1.1 | Request、统一 Scheduler、Engine control loop、Executor contract | 已完成 |
| V1.2 | 单类型 full-attention 逻辑 Paged KV Cache、prefix cache、preemption/recompute | 主体已完成 |
| V1.3 | Physical KV Cache、reference full attention、Tiny Llama、ModelRunner、Worker、SingleProcessExecutor | 当前阶段 |
| V1.4 | safetensors 权重加载、真实 Llama/Qwen checkpoint、完整 sampling 和离线 API | 待实现 |
| V1.5 | sliding window、chunked-local、多 KV cache group、Mistral/hybrid model | 延后实现 |
| V1.6 | optimized attention、CUDA Graph/compile、benchmark 和性能收尾 | 待实现 |

详细执行清单：

- [V1.1 Control Loop](../implementation/v1-control-loop-todo.md)
- [V1.2 Paged KV Cache](../implementation/v1-paged-kv-cache-todo.md)
- [V1.3 Model Execution](../implementation/v1-model-execution-todo.md)

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
- [ ] 支持 max model length 的 admission、调度和终止语义。

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
- [ ] 实现真实 `SingleProcessExecutor`。

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
- [ ] 调度同时遵守 max model length。

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
- [ ] 在 ModelRunner 中建立唯一的 token position → logical block → physical slot helper。

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
- [ ] 使用真实模型验证 prefix cache on/off 输出一致。
- [ ] 使用真实模型验证 preempt/recompute 与无抢占输出一致。

## 6. V1.3：Full-Attention Model Execution（当前阶段）

### Physical KV Cache

- [ ] Worker 为每个 decoder layer 创建 physical KV cache tensor。
- [ ] KV tensor 的 block dimension 与 CacheConfig physical block ID 一一对应。
- [ ] tensor shape 表达 K/V、block、block offset、KV heads 和 head dim。
- [ ] physical cache dtype/device 由 Worker 配置决定。
- [ ] Worker 拥有 tensor，Scheduler/KVCacheManager 不访问 tensor。
- [ ] block 被逻辑释放后不强制清零，slot overwrite 和 block table 保证隔离。

### ModelInput 与 AttentionMetadata

- [ ] 将 SchedulerOutput 转换为 flattened input IDs。
- [ ] 生成每个 token 的 absolute position。
- [ ] 使用唯一 helper 生成 slot mapping。
- [ ] 将不同长度 block tables pad 为只读 metadata tensor。
- [ ] 生成 query start locations 和每 request context length。
- [ ] 生成 sampling row indices。
- [ ] 支持 request 具有不同 scheduled token 数的 mixed batch。
- [ ] 不建立 prefill/decode 两套 ModelInput。

### Reference Full Paged Attention

- [ ] 将当前 query token 的 K/V 写入 slot mapping 指定位置。
- [ ] 根据 request block table 读取完整 causal history。
- [ ] chunked prefill 同时读取历史 KV 和当前 chunk。
- [ ] decode 读取完整 full-attention context。
- [ ] 支持 MHA、GQA 和 MQA。
- [ ] softmax 使用数值稳定 dtype。
- [ ] reference backend 不访问 Scheduler、Request 或 KVCacheManager。
- [ ] full prefill、chunked prefill 和 decode 与 dense causal reference 对拍。

### Tiny Llama

- [ ] 定义最小 Llama architecture config，不建立万能 model config。
- [ ] 实现 RMSNorm。
- [ ] 实现标准 Llama RoPE。
- [ ] 实现单卡 dense Linear、Embedding 和 LM head。
- [ ] 实现 SwiGLU MLP。
- [ ] 实现 Q/K/V/O projection 和 full attention layer。
- [ ] 实现 decoder layer residual/pre-norm 结构。
- [ ] 实现多层 Llama model 和 final RMSNorm。
- [ ] 模型 `forward()` 与 `compute_logits()` 分离。
- [ ] 与 Transformers tiny Llama 的 layer、hidden state 和 logits 对拍。

### Sampler、ModelRunner、Worker 和 Executor

- [ ] 实现 greedy sampler。
- [ ] sampler 输出通过 request ID 显式关联。
- [ ] ModelRunner 构造统一 flattened batch 和 attention metadata。
- [ ] ModelRunner 只对 sampling rows 计算/收集 logits。
- [ ] ModelRunner 不修改 Request、SchedulerOutput 或 Scheduler 状态。
- [ ] Worker 拥有 device、model、physical KV cache 和 ModelRunner。
- [ ] Worker 提供显式 shutdown。
- [ ] 实现 SingleProcessExecutor 并只委托 Worker。
- [ ] Engine 使用真实 SingleProcessExecutor 完成 tiny-model generation。

### 执行正确性

- [ ] 一次 full prefill 与多次 chunked prefill 的最后 logits/token 一致。
- [ ] prefill + decode 与完整序列 forward 一致。
- [ ] mixed decode/prefill batch 与逐 request 独立执行一致。
- [ ] prefix cache on/off 输出一致。
- [ ] preempt/recompute 与无抢占输出一致。
- [ ] physical block 释放和复用后不存在 stale KV 污染。
- [ ] CPU FP32 reference tests 通过。
- [ ] 单 GPU eager tests 通过。

## 7. V1.4：权重加载、模型覆盖与完整 Sampling

### Model Registry 和模型接口

- [ ] Model Registry 根据 Hugging Face `architectures` 选择实现。
- [ ] 模型类负责架构差异，不把模型判断放入 Scheduler/ModelRunner。
- [ ] 模型 `forward()`、`compute_logits()` 和 `load_weights()` 职责分离。
- [ ] 公共 RMSNorm、RoPE、MLP、Embedding、Linear 和 Attention layer 可复用。
- [x] 不实现难以理解的万能声明式 Transformer builder。

### 权重加载

- [ ] 支持单文件 safetensors。
- [ ] 支持 safetensors index 和多 shard 流式加载。
- [ ] 避免同时将全部 checkpoint tensor 常驻 CPU 内存。
- [ ] 支持 tied embedding/lm_head。
- [ ] 支持 QKV packed parameter 映射。
- [ ] 支持 gate/up packed parameter 映射。
- [ ] module/parameter 可以注册自己的 weight loader。
- [ ] loader 接受 size=1 的显式 ParallelContext 或等价 shard context。
- [ ] missing、unexpected、shape mismatch 权重给出明确错误。

### 模型覆盖

- [ ] 加载真实 Llama-family dense checkpoint。
- [ ] 加载真实 Qwen2/Qwen3 dense checkpoint。
- [ ] Llama 和 Qwen greedy 输出与 Transformers 逐 token 对拍。
- [ ] 至少两个真实小型模型族完成端到端生成。

### Sampling 和离线 API

- [ ] greedy decoding。
- [ ] temperature sampling。
- [ ] top-k sampling。
- [ ] top-p sampling。
- [ ] per-request random generator/seed。
- [x] EOS、ignore EOS 和 max tokens 控制流。
- [ ] max model length admission 和终止原因。
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

### 待补测试

- [ ] mixed batch 与逐 request 真实模型执行结果一致。
- [ ] prefix cache on/off 真实模型输出一致。
- [ ] preempt/recompute 与无抢占真实模型输出一致。
- [ ] full/sliding/chunked-local 与 dense reference 对拍。
- [ ] Llama、Qwen 和 Mistral 最小权重端到端加载测试。
- [ ] greedy 输出与 Transformers 逐 token 对拍。

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
- [ ] decode 与 chunked prefill 在同一个真实模型 batch 中正确执行。
- [x] 任意 Scheduler step 严格遵守 token 和 sequence budget。
- [ ] KV 空间不足时能抢占并最终正确完成真实模型请求，无 OOM 或 block 泄漏。
- [ ] prefix cache on/off 真实模型输出一致。
- [ ] full、sliding window 和 chunked-local 均有独立 reference correctness test。
- [ ] reference 与 optimized attention 通过数值对拍。
- [ ] 核心对象职责、不变量和一次完整 request trace 已文档化。
- [ ] benchmark 记录 throughput、TTFT、ITL、显存和 KV utilization。
