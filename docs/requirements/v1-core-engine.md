# V1：单卡 Core Engine 需求清单

## 1. 版本目标

在单 GPU 上建立完整、可验证的 LLM serving 核心，形成后续分布式、MoE、量化、投机解码和多模态版本共同依赖的稳定控制流。

V1 必须回答以下问题：

- 请求如何从 waiting 进入 running 并最终结束；
- 如何按照 token budget 将 prefill 和 decode 放入同一个 batch；
- KV cache 如何按 block 分配、共享、释放和抢占；
- Scheduler 如何生成与设备执行解耦的执行计划；
- ModelRunner 如何把计划转换为 GPU 输入；
- 模型如何声明 attention/KV 需求并加载 Hugging Face 权重。

## 2. 建议内部里程碑

- **V1.1 正确性闭环**：统一调度、单进程执行、dense model、reference attention。
- **V1.2 Paged KV Cache**：block pool、block table、prefix cache、preemption。
- **V1.3 Attention Pattern**：full、sliding window、chunked local 和 cache group。
- **V1.4 Optimized Backend**：代表性的 GPU prefill/decode backend 和性能基线。

## 3. Engine 与请求生命周期

### P0

- [ ] `LLM` 提供离线批量 `generate()` 接口。
- [x] Engine 提供逐 step 驱动接口，允许上层实现 streaming。
- [x] 支持动态添加请求，而非只能静态处理一次性 batch。
- [x] Request 包含稳定 ID、prompt tokens、output tokens、sampling params 和状态。
- [x] 请求状态包含 `WAITING`、`RUNNING`、`FINISHED`；抢占后回到 `WAITING`。
- [x] `num_computed_tokens` 是请求计算进度的唯一权威字段。
- [x] Engine 负责调用 `schedule -> execute -> update`，三阶段职责清晰。
- [ ] 请求结束时释放全部非共享 KV block。
- [ ] 支持 abort/cancel 请求并安全释放资源。
- [ ] 明确处理空 prompt、超长 prompt、达到 max model length 等边界情况。

### P1

- [ ] 提供异步迭代器形式的轻量 streaming API。
- [ ] 支持按请求返回累计输出和增量输出。

## 4. 统一 Scheduler

### P0

- [x] Scheduler 同时维护 waiting 和 running 请求，但不维护独立 prefill/decode 执行路径。
- [x] 每一步以 `max_num_batched_tokens` 作为全局 token budget。
- [x] 每一步以 `max_num_seqs` 作为请求数上限。
- [x] Scheduler 为每个请求输出 `num_scheduled_tokens`。
- [x] running 请求优先于新 waiting 请求，避免 decode 被长 prompt 长时间阻塞。
- [x] 在剩余 budget 内调度 waiting 请求并执行 chunked prefill。
- [x] 长 prompt 可以跨多个 step 完成，且中间 step 不产生采样 token。
- [x] 同一个 batch 可以混合 decode token 和 prefill chunk。
- [x] budget 不足时，prefill chunk 必须截断而不是整请求阻塞。
- [ ] 每请求计划 token 数不得超过其 pending token 数和 max model length。
- [x] 只有当请求计算追上最后一个已知 token 时才进行采样。
- [ ] SchedulerOutput 不包含 GPU tensor，只包含控制信息和 block ID。
- [x] SchedulerOutput 创建后视为不可变，Executor 不得修改 Scheduler 内部状态。

### 调度策略

- [x] V1 默认采用 FIFO admission。
- [x] running 列表保持稳定顺序，保证行为易复现。
- [ ] 资源不足时优先抢占最近进入或最低优先级的请求。
- [ ] 抢占策略封装为独立 policy，不能散落在 Scheduler 主循环中。
- [ ] 调度决策写入 trace，包括未被选择请求的原因。

### P1

- [ ] 支持配置单请求最大 prefill chunk。
- [ ] 支持简单 request priority，但不要求复杂公平性策略。

## 5. Paged KV Cache

### BlockPool

- [ ] 管理固定数量的 physical block ID。
- [ ] 使用确定性的 free queue，便于测试和 trace。
- [ ] 每个 block 维护 refcount。
- [ ] refcount 为零时 block 才能回到 free queue。
- [ ] 检测 double free、负 refcount、重复分配等错误。
- [ ] Engine 侧 BlockPool 不持有实际 KV tensor。

### BlockTable

- [ ] 每个请求维护 logical block 到 physical block 的映射。
- [ ] 支持追加 block、查询 slot、截断和完整释放。
- [ ] 支持用 null/unmapped block 表示 local attention 已经淘汰的逻辑位置。
- [ ] block table 的逻辑 token 范围与 block size 计算只有一个实现来源。

### PrefixCache

- [ ] 对完整 token block 计算稳定 hash。
- [ ] hash 必须包含前序 hash，避免只按当前 block 内容误共享。
- [ ] hash 可包含 model/cache group 等隔离信息。
- [ ] 只有完整且计算完成的 block 可以进入 prefix cache。
- [ ] partial block 在 V1 不共享，不实现 copy-on-write。
- [ ] cache hit 时增加 physical block refcount。
- [ ] block 被复用前验证 hash 和 cache group 一致。
- [ ] 请求结束后，共享 block 只有 refcount 为零才可释放。

### KVCacheManager

- [ ] 提供 `allocate_slots(request, num_tokens)`。
- [ ] 提供 `find_prefix_hit(request)`。
- [ ] 提供 `free_request(request)`。
- [x] preempt request 释放 cache 并回到 `WAITING`，重新调度后 recompute。
- [ ] allocation 是原子的：失败时不得遗留半分配 block。
- [x] preemption 采用释放 KV、恢复后 recompute 的策略。
- [x] 恢复时允许重新命中 prefix cache。
- [x] 不实现 CPU swap。

## 6. Attention 与 KV Cache Spec

### P0

- [ ] 定义不可变 `AttentionSpec` 基类或协议。
- [ ] 实现 `FullAttentionSpec`。
- [ ] 实现 `SlidingWindowSpec(window_size)`。
- [ ] 实现 `ChunkedLocalSpec(chunk_size)`。
- [ ] 明确定义每种 spec 的 query-key 可见范围。
- [ ] 每个 attention layer 声明自己的 spec，不由 Scheduler 推断模型类型。
- [ ] 相同 spec 的 layer 可以组成 `KVCacheGroup`。
- [ ] 每个请求为每个 cache group 持有独立 block table。
- [ ] full 与 local layer 混合时，local group 可以回收旧 block，full group 必须保留完整历史。
- [ ] cache manager 与 attention mask 使用同一份 spec 参数。

### Reference Backend

- [ ] 使用 PyTorch 实现 full causal attention reference。
- [ ] 使用显式 mask 实现 sliding window reference。
- [ ] 使用显式 mask 实现 chunked local reference。
- [ ] reference backend 能读取 logical block table 对应的 paged KV。
- [ ] reference backend 支持 prefill、chunked prefill 和 decode。

### Optimized Backend

- [ ] 至少提供一个 GPU prefill 实现。
- [ ] 至少提供一个 paged decode attention 实现。
- [ ] chunked prefill 能同时读取历史 paged KV 和当前 chunk。
- [ ] optimized backend 与 reference backend 在容许误差内对齐。

### P1

- [ ] 对 optimized backend 增加 CUDA Graph 或 `torch.compile` 实验开关。
- [x] 优化项不得进入 Scheduler 核心抽象。

## 7. Executor、Worker 与 ModelRunner

### Executor

- [x] 定义稳定 Executor protocol。
- [ ] 实现 `SingleProcessExecutor`。
- [ ] Executor 负责初始化 Worker、探测可用显存、初始化 KV cache 和执行 plan。
- [x] Engine 不直接访问 CUDA device 或 model module。

### Worker

- [ ] Worker 拥有 device、model weights、KV tensors 和 ModelRunner。
- [ ] Worker 根据 cache config 一次性初始化物理 KV tensor。
- [ ] physical block ID 与 Engine 侧 BlockPool ID 对齐。
- [ ] Worker 提供明确 shutdown，释放设备资源。

### ModelRunner

- [ ] 将 SchedulerOutput 转换为 input IDs、positions、slot mapping 和 block tables。
- [ ] 支持不同请求、不同 scheduled token 数的 flattened batch。
- [ ] 生成 query/request 边界和 attention metadata。
- [ ] 只为达到采样位置的请求计算或收集 logits。
- [ ] Sampler 输出与 request ID 明确对应，不依赖隐含 batch 顺序。
- [ ] ModelRunner 不修改 Request 和 Scheduler 状态。

## 8. 模型与权重加载

### 模型抽象

- [ ] Model Registry 根据 Hugging Face `architectures` 选择模型实现。
- [ ] 模型实现可返回逐层 `AttentionSpec`。
- [ ] 模型 `forward()`、`compute_logits()` 和 `load_weights()` 职责分离。
- [ ] 公共 RMSNorm、RoPE、MLP、Embedding、Linear 和 Attention layer 可复用。
- [x] 不实现难以理解的万能声明式 Transformer builder。

### 模型覆盖

- [ ] 支持 Llama dense 模型族，验证 full attention。
- [ ] 支持 Qwen2/Qwen3 dense 模型族，验证 packed weights 和结构差异。
- [ ] 支持 Mistral 或等价模型，验证 sliding window。
- [ ] 使用 synthetic tiny model 验证 full + chunked local hybrid，不要求 V1 加载完整 Llama 4。

### 权重加载

- [ ] 支持单文件 safetensors。
- [ ] 支持 safetensors index 和多 shard 流式加载。
- [ ] 避免同时将全部 checkpoint tensor 常驻 CPU 内存。
- [ ] 支持 tied embedding/lm_head。
- [ ] 支持 QKV packed parameter 映射。
- [ ] 支持 gate/up packed parameter 映射。
- [ ] module/parameter 可以注册自己的 weight loader。
- [ ] loader 接受 `ParallelContext`，V1 中其 rank/size 均为单卡值。
- [ ] 对 missing、unexpected、shape mismatch 权重给出明确错误。

## 9. Sampling

### P0

- [ ] greedy decoding。
- [ ] temperature sampling。
- [ ] top-k sampling。
- [ ] top-p sampling。
- [ ] per-request random generator/seed。
- [x] EOS、ignore EOS 和 max tokens。
- [ ] max model length 终止原因。

### 非目标

- [x] 不实现 beam search。
- [x] 不实现 best-of 和 parallel sampling。
- [x] 不实现 structured/guided decoding。
- [x] 不实现用户自定义 logits processor。

## 10. 可观察性与测试

### Trace

- [ ] 每一步记录总 token budget 和已使用 budget。
- [ ] 记录每个请求 scheduled token 数。
- [ ] 记录请求 admission、preemption、resume 和 finish。
- [ ] 记录 block allocate/free/share 和 refcount 变化。
- [ ] 记录 prefix hit token/block 数。
- [ ] 记录 batch 中 prefill token 与 decode token 的构成。

### 测试

- [x] Scheduler 永不超过 token 和 sequence budget。
- [x] 长 prompt 能被正确切成多个 chunk。
- [ ] mixed batch 与分开执行结果一致。
- [ ] prefix cache on/off 输出一致。
- [ ] preempt/resume 与不抢占输出一致。
- [ ] BlockPool 随机操作 property test 无泄漏。
- [ ] full/sliding/chunked local 与 dense mask reference 对拍。
- [ ] 不同 block size 和 chunk size 下结果一致。
- [ ] Llama、Qwen 和 Mistral 最小权重端到端加载测试。
- [ ] greedy 输出与 Transformers 逐 token 对拍。

## 11. 明确不包含

- 多 GPU 和多进程 Worker；
- CPU KV swap/offload；
- MoE；
- 权重或 KV 量化；
- speculative decoding；
- multimodal input；
- LoRA；
- OpenAI API server；
- Ray、RPC framework、故障恢复；
- 多套 optimized attention backend。

## 12. V1 发布验收

- [ ] 单 GPU 可加载至少两个真实的小型模型族并生成文本。
- [ ] decode 与 chunked prefill 可出现在同一个执行 batch。
- [ ] 任意 step 均严格遵守 token budget。
- [ ] KV 空间不足时能抢占并最终正确完成请求，不发生 OOM 或 block 泄漏。
- [ ] prefix cache、sliding window 和 chunked local 均有独立正确性测试。
- [ ] reference 与 optimized attention 通过数值对拍。
- [ ] 核心对象职责、不变量和一次完整 request trace 已文档化。
- [ ] V1 benchmark 记录 throughput、TTFT、ITL、显存和 KV utilization。
