# V1.5 多 Attention 语义与多 KV Cache Group 实现清单

## 1. 目标

在不改变 Request 进度模型、token-budget Scheduler 和 Engine control loop 的
前提下，将当前“全部 layer 共享一个 full-attention cache group”扩展为：

```text
model layers
    -> immutable AttentionSpec per layer
    -> layers with the same cache semantics form one KVCacheGroup
    -> Scheduler reserves every group atomically
    -> ScheduledRequest carries one block-table snapshot per group
    -> ModelRunner builds one AttentionMetadata per group
    -> each layer selects its group's metadata and physical KV tensor
```

V1.5 完成后必须支持：

- full causal attention；
- sliding-window attention；
- chunk-aligned local attention；
- 同一模型内 full + local 的 hybrid layer 组合；
- 各 cache group 独立分配、回收、抢占和释放 block；
- Mistral-family sliding-window 模型的 checkpoint 加载和端到端执行。

## 2. 范围和非目标

### 本阶段实现

- [ ] `FullAttentionSpec`、`SlidingWindowSpec` 和 `ChunkedLocalSpec`。
- [ ] 将 attention 语义与 operator 实现继续保持分离。
- [ ] 将逻辑 BlockPool/BlockTable 扩展为多 cache group。
- [ ] bounded-context group 在成功 update 后回收已不可见的旧 block。
- [ ] SchedulerOutput、ModelInput、AttentionMetadata 和 physical KV cache 支持多 group。
- [ ] synthetic full + chunked-local hybrid model 端到端执行。
- [ ] tiny Mistral 与 Transformers dense-mask/reference 对拍。
- [ ] 选定一个可在当前 T4 环境验收的固定公开 Mistral-family checkpoint。

### 本阶段不实现

- optimized/CUDA attention operator；仍使用 reference operator。
- 不同 cache group 使用不同 block size、dtype 或 block capacity。
- cyclic/ring block table；首版保留 absolute logical block index 和稀疏 `None` mapping。
- CPU KV swap/offload、KV connector 或 disaggregated serving。
- 对 non-full/hybrid cache group 启用 prefix cache。
- 为了模型覆盖建设通用声明式 Transformer builder。
- 修改 Scheduler 的 token budget、FIFO、victim selection 或 recompute 策略。

## 3. 核心语义

### 3.1 AttentionSpec

`AttentionSpec` 是不含 tensor/device 的 frozen value object。对 query absolute
position `p`，可见 key 范围统一定义为闭区间：

```text
context_start(p) <= key_position <= p
```

| Spec | `context_start(p)` | 语义 |
|---|---|---|
| `FullAttentionSpec` | `0` | 保留完整 causal history |
| `SlidingWindowSpec(window_size)` | `max(0, p - window_size + 1)` | 包含当前 token 在内最多 `window_size` 个 token |
| `ChunkedLocalSpec(chunk_size)` | `(p // chunk_size) * chunk_size` | 只查看当前对齐 chunk 内的 causal prefix |

不将 window/chunk 语义编码到 operator 名称、Scheduler 分支或模型类型
判断中。`Attention` 拥有 spec 并据此构造
`query_context_start_positions`；`AttentionOperator` contract 保持不变。

### 3.2 KVCacheGroupSpec 与 KVCacheLayout

在 `core/kv_cache_spec.py` 中定义共享的不可变布局：

```python
@dataclass(frozen=True, slots=True)
class KVCacheGroupSpec:
    group_id: int
    attention_spec: AttentionSpec
    layer_indices: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class KVCacheLayout:
    block_size: int
    num_blocks_per_group: int
    groups: tuple[KVCacheGroupSpec, ...]
```

约束：

- `group_id` 从 `0` 开始连续编号，按模型 layer 首次出现顺序稳定生成。
- 每个 layer 恰好属于一个 group，`layer_indices` 不重复且完整覆盖。
- 同一 group 中的 layer 必须具有相同 `AttentionSpec`。
- V1.5 所有 group 使用相同 block size 和 `num_blocks_per_group`。
- block ID 只在 group 内有效；`group 0 / block 3` 与 `group 1 / block 3`
  是两块独立 physical storage。

Model 不导入 `core` 或构造 `KVCacheLayout`。模型只暴露：

```text
attention_groups: tuple[Attention, ...]
layer_attention_group_ids: tuple[int, ...]
```

ModelRunner 校验模型声明并构造 `KVCacheLayout`，Worker 只读暴露该
layout 给 control plane。

### 3.3 SchedulerOutput

将单个 `ScheduledRequest.block_table` 替换为显式 group snapshot：

```python
@dataclass(frozen=True, slots=True)
class CacheGroupBlockTable:
    group_id: int
    block_table: tuple[int | None, ...]

@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    ...
    cache_groups: tuple[CacheGroupBlockTable, ...]
```

- tuple 顺序与 `KVCacheLayout.groups` 一致。
- plan 不携带 `Attention`、`AttentionSpec`、BlockPool 或 device tensor。
- Executor/ModelRunner 不允许修改 snapshot。
- 不保留单 group 特殊字段；full-only 模型也统一使用一元 tuple。

### 3.4 逻辑 block 分配和回收

`KVCacheManager` 拥有每 group 独立的 BlockPool 和 request BlockTable。
对一个 request 的分配必须是跨 group 原子操作：

1. 计算每 group 当前 scheduled positions 需要的 absolute logical blocks。
2. 对所有 BlockPool 执行无副作用 capacity preflight。
3. 只有所有 group 都可容纳时才提交 acquire/allocate。
4. 任一 group 容量不足时返回失败，不改变任何 pool/table。

bounded-context group 只在 `Scheduler.update()` 成功后回收旧 block：

```text
next_query_position = request.num_computed_tokens
first_visible_position = spec.context_start(next_query_position)
first_retained_logical_block = first_visible_position // block_size
```

所有 logical index 小于 `first_retained_logical_block` 的 mapping 被替换为
`None` 并释放 refcount。包含 visibility boundary 的 block 整块保留，额外保留
的 block-prefix token 由 attention mask 隔离。

plan retry 之前、executor 失败时或尚未 update 时不得回收 block。preempt、
abort 和 finish 必须一次性释放 request 在所有 group 的引用。

### 3.5 Prefix cache 边界

V1.5 首版保持明确限制：

- 单一 `FullAttentionSpec` group 继续支持现有 prefix cache。
- 当 layout 包含 sliding/chunked-local group 或多个 group 时，
  `enable_prefix_cache=True` 必须在初始化阶段明确拒绝。
- `hash_block()` 保留 `cache_group_id` 作为未来 group-aware prefix cache 的
  identity 边界，但本阶段不实现 partial-window prefix resume。

不允许通过静默关闭 prefix cache 隐藏用户配置错误。

## 4. Phase 0：冻结 contract 和不变量

- [x] 新增 `attention/spec.py`，只保存 immutable semantic values 和 domain 校验。
- [x] 定义三种 spec 的精确可见区间，特别锁定 window 是否包含当前 token。
- [x] 新增 `KVCacheGroupSpec` 和 `KVCacheLayout`。
- [x] 定义 model 暴露 attention group 的最小 runner contract。
- [x] 定义稳定 group ID、layer 覆盖、block ID namespace 和 tuple ordering。
- [x] 内部 group/layout 依赖类型与构造流程保证不变量，不做重复运行时校验；
      仅拒绝来自模型配置的非正 window/chunk size。
- [x] 记录 full-only 现有模型如何映射为单 group，不保留另一条运行路径。

## 5. Phase 1：Attention 语义和 reference correctness

- [ ] `Attention` 构造时显式接收 `AttentionSpec` 和 operator name。
- [ ] full spec 生成全零 `query_context_start_positions`。
- [ ] sliding spec 在 prefill、chunked prefill 和 decode 下生成正确 start positions。
- [ ] chunked-local spec 在 query chunk 跨越 local boundary 时为每个 query 生成独立 start。
- [ ] `AttentionOperator` 签名不增加 spec/model/scheduler 参数。
- [ ] reference gather 只读取当前 batch 所需的最小连续 physical context。
- [ ] full、sliding 和 chunked-local 分别与显式 dense boolean mask 对拍。
- [ ] 覆盖 window/chunk 小于、等于和大于 block size 的组合。
- [ ] 覆盖 MHA、GQA、多 request mixed batch 和跨 boundary chunked prefill。
- [ ] 现有 full-attention 数值对拍不回归。

## 6. Phase 2：多 Group 逻辑 KV Cache

- [ ] `KVCacheManager` 按 layout 创建每 group 独立 BlockPool。
- [ ] 每 request 按 group 拥有独立 BlockTable。
- [ ] BlockTable 支持将 absolute logical block mapping 替换为 `None` 并返回旧 ID。
- [ ] BlockTable 支持为指定 absolute logical indices 填充缺失 mapping。
- [ ] BlockPool 提供无副作用 capacity preflight。
- [ ] 多 group allocation 成功时一次性返回全部 immutable snapshots。
- [ ] 任一 group 容量不足时不改变任何 free queue、refcount 或 table。
- [ ] `num_free_blocks` 不再隐含表示唯一 pool，对外提供按 group 的 snapshot。
- [ ] sliding group 在 window 向前移动后释放完全不可见 block。
- [ ] chunked-local group 进入新 local chunk 后释放上一 chunk block。
- [ ] visibility boundary 所在的 partial block 保留，mask 隔离 block 内额外 token。
- [ ] plan retry 不重复分配，failed update 不提前回收。
- [ ] preempt/recompute、abort、finish 和 repeated free 覆盖所有 group。
- [ ] 抢占 victim 仍由 Scheduler 选择，manager 只提供分配/释放机制。

## 7. Phase 3：SchedulerOutput 和 ModelInput

- [ ] 引入 frozen `CacheGroupBlockTable`。
- [ ] `ScheduledRequest` 统一携带 group snapshots，移除单 `block_table` 特例。
- [ ] Scheduler 只请求“跨 group 原子分配”，不解释 spec 类型。
- [ ] allocation 失败时仍沿用现有 deterministic victim selection。
- [ ] `build_block_table()` 按 group 生成 padded tensor。
- [ ] `build_slot_mapping()` 按 group 生成 absolute position 到 physical slot 的映射。
- [ ] `ModelInput` 显式保存 common positions 和 metadata tuple，不从 group 0 隐式推导。
- [ ] 每 group 共享 input IDs/query row ordering，但使用自己的 block table、slot mapping 和 context starts。
- [ ] sampling row 和 request ID 关联不受 group 数量影响。
- [ ] SchedulerOutput 仍不含 tensor、Request、model object 或 mutable container。

## 8. Phase 4：ModelRunner 和 Physical KV Cache

- [ ] runner model protocol 增加 `attention_groups` 和 `layer_attention_group_ids`。
- [ ] ModelRunner 校验 group/layer mapping 并生成 immutable `KVCacheLayout`。
- [ ] Worker 只读暴露 runner 初始化后的 layout，不自行构造 spec。
- [ ] `LLM` 在 Worker 初始化后用 layout 构造 KVCacheManager/Scheduler。
- [ ] warmup plan 为每 group 创建合法 block-table snapshot。
- [ ] GPU profile 仍将 `num_blocks` 解释为每 group 的相同 capacity。
- [ ] bytes-per-block 统计一个 block index 在所有 group/layer 的总 K/V 字节。
- [ ] PhysicalKVCache 仍以 layer 顺序暴露 tensor，但每层只使用所属 group 的 block namespace。
- [ ] model forward 按 `layer_attention_group_ids` 选择 metadata，不在 layer 内查全局 registry。
- [ ] shutdown 释放所有 physical layer cache，不留 group-specific tensor 引用。
- [ ] CPU deterministic capacity 和 CUDA profiled capacity 均覆盖单/多 group。

## 9. Phase 5：Hybrid Model 端到端验收

- [ ] 构造 test-only tiny model，layer 按固定顺序交替 full 和 chunked-local。
- [ ] 模型显式声明两个 attention group 和 layer-to-group mapping。
- [ ] 一次 full prefill 与多次 chunked prefill 的最终 hidden state/logits 一致。
- [ ] prefill + decode 与显式 dense per-layer mask reference 对拍。
- [ ] mixed request batch 与逐 request 独立执行一致。
- [ ] local group 回收旧 block 时 full group 仍保留完整历史。
- [ ] local group 的 physical block ID 复用不产生 stale KV 污染。
- [ ] 容量压力下任一 group 失败均触发完整 request preemption/recompute。
- [ ] finish/abort 后所有 group refcount 归零，free queue 恢复预期顺序。
- [ ] hybrid layout 启用 prefix cache 时立即返回可操作错误。

## 10. Phase 6：Mistral-family 集成

### 模型实现

- [ ] 完成 `models/mistral.py`，不在 Loader/Runner/Scheduler 中添加 Mistral 分支。
- [ ] 从 `ModelConfig` 读取 hidden/intermediate size、layer/head 数、RoPE、norm eps 和 sliding window。
- [ ] 明确拒绝当前不支持的 rope scaling、attention bias 或其他改变语义的配置。
- [ ] 所有 decoder layer 声明同一 `SlidingWindowSpec`。
- [ ] 复用已有 RMSNorm、RoPE、SwiGLU、Embedding/LMHead 和 Linear。
- [ ] 只在结构完全相同时复用 Llama decoder 小组件，不建立通用 model builder。
- [ ] `load_weights()` 拥有 Mistral checkpoint name mapping 和 tied-weight 语义。
- [ ] registry 增加 `MistralForCausalLM` 静态映射。

### 数值与端到端

- [ ] 使用 Transformers tiny `MistralConfig` 生成确定性 HF-format checkpoint。
- [ ] 参数、hidden states 和 logits 与 Transformers eager reference 对拍。
- [ ] full prefill、chunked prefill 和多 token decode 逐 token 一致。
- [ ] 验证 sliding window 跨 block boundary 和跨 prefill chunk boundary。
- [ ] Engine 路径验证 block reclamation、reuse、preemption 和 shutdown。
- [ ] 选定公开 checkpoint 前记录 repo、commit、权重大小、dtype 和 T4 memory budget。
- [ ] 公开 checkpoint 首次下载后支持 `local_files_only=True` 离线重载。
- [ ] 固定 raw prompt/token IDs，与 Transformers 完成 logits 和 greedy token 对拍。

## 11. Phase 7：回归、可观察性和文档

- [ ] 现有 Llama/Qwen full-only 模型统一走单 group 新路径。
- [ ] 现有 prefix cache on/off、preemption、block reuse 和 loaded model 对拍不回归。
- [ ] 结构化调试输出能按 group 报告 spec、layer indices、block table 和 free capacity。
- [ ] 记录 bounded group 每 step 回收的 block/token 数。
- [ ] 记录 physical cache 的 group/layer shape、dtype 和总字节。
- [ ] 更新 `docs/requirements/v1-core-engine.md` 的 V1.5 勾选状态。
- [ ] 更新 `docs/requirements/README.md` 中 Attention/KV cache 稳定 contract。
- [ ] 增加 full、sliding 和 hybrid 的最小数据流示意。
- [ ] 记录 prefix cache 在 bounded/hybrid layout 中的明确限制。

## 12. 测试分层

| 层级 | 目录/文件 | 重点 |
|---|---|---|
| Spec unit | `tests/attention/test_specs.py` | 参数校验和 context-start 语义 |
| Attention reference | `tests/attention/operators/test_reference.py` | sliding/local 与 dense mask 对拍 |
| Logical cache | `tests/core/test_kv_cache_manager.py` | 多 group 原子分配、回收和释放 |
| Scheduler | `tests/core/test_scheduler.py` | 多 group snapshot、capacity 失败和抢占 |
| Runner | `tests/worker/test_model_runner_execution.py` | metadata tuple、physical layout 和 warmup/profile |
| Hybrid integration | `tests/integration/test_hybrid_attention_engine.py` | full + local 端到端 |
| Mistral model | `tests/models/test_mistral_model.py` | config、权重和 Transformers 数值对拍 |
| Mistral engine | `tests/integration/test_loaded_mistral_engine.py` | checkpoint、chunk/decode、回收和复用 |
| Public model | `tests/integration/test_hub_mistral.py` | pinned Hub GPU/offline 验收 |

测试不得只检查“运行不报错”或“生成文本看起来合理”。至少对比
visible key positions、hidden/logits/token IDs 和 block lifecycle 中的一项确定性
reference。

## 13. 预计文件变更

```text
src/micro_vllm/attention/spec.py                 # new: immutable attention semantics
src/micro_vllm/attention/attention.py            # spec-driven metadata
src/micro_vllm/core/kv_cache_spec.py             # group/layout contracts
src/micro_vllm/core/block.py                     # sparse absolute mappings/preflight
src/micro_vllm/core/kv_cache_manager.py          # multi-group allocation/retention
src/micro_vllm/core/scheduler_output.py           # per-group snapshots
src/micro_vllm/core/scheduler.py                  # consume atomic group allocation
src/micro_vllm/worker/kv_cache.py                 # group-aware physical layout
src/micro_vllm/worker/model_runner.py             # metadata tuple/layout exposure
src/micro_vllm/worker/worker.py                   # expose initialized layout
src/micro_vllm/llm.py                             # pass layout to control plane
src/micro_vllm/models/llama.py                    # declare one full group
src/micro_vllm/models/qwen.py                     # declare one full group
src/micro_vllm/models/mistral.py                  # sliding-window model
src/micro_vllm/models/registry.py                 # Mistral static mapping
```

如果实现过程需要修改 Engine、Request 或 Sampler contract，应先停止并复查
设计；正常的 V1.5 扩展不需要改变这三者。

## 14. 远端验证命令

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q \
   tests/attention tests/core/test_kv_cache_manager.py tests/core/test_scheduler.py'

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q \
   tests/worker tests/integration/test_hybrid_attention_engine.py'

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q \
   tests/models/test_mistral_model.py tests/integration/test_loaded_mistral_engine.py'

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m "not model"'

ssh aliyun-ecs \
  'cd /root/micro-vllm && MICRO_VLLM_TEST_MODEL=1 HF_HUB_OFFLINE=1 \
   .venv/bin/python -m pytest -q -m "model and gpu" \
   tests/integration/test_hub_mistral.py'

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q'
```

公开 Mistral checkpoint 第一次下载时去掉 `HF_HUB_OFFLINE=1`；确认标准
Hub cache 完整后再执行上述离线命令。公开模型测试应单独运行，避免
与其他大模型在同一进程中叠加 GPU/CPU memory。

## 15. 建议提交顺序

1. `feat: define attention and cache group specs`
2. `feat: add sliding and chunked-local reference attention`
3. `feat: manage multiple logical kv cache groups`
4. `refactor: carry cache groups through execution plans`
5. `feat: execute hybrid attention cache groups`
6. `feat: add mistral sliding-window model`
7. `test: validate pinned mistral checkpoint`
8. `docs: complete V1.5 multi-attention milestone`

每个提交应保持可读、可回归的中间状态；不将 spec、cache manager、runner
和 Mistral 全部压入一个提交。

## 16. 完成标准

- [ ] FullAttention 现有测试和 Llama/Qwen 端到端结果不变。
- [ ] SlidingWindow 和 ChunkedLocal 都有独立 dense-mask reference 对拍。
- [ ] Scheduler 仍只根据 token/sequence budget 和分配成败调度，不读取模型类型。
- [ ] 任一 cache group 分配失败都不会留下部分 reservation。
- [ ] bounded group 能回收旧 block，full group 仍保留完整历史。
- [ ] hybrid model 通过 prefill、chunked prefill、decode、mixed batch 和 preemption 验收。
- [ ] Mistral-family 通过 local tiny checkpoint 和固定公开 checkpoint 验收。
- [ ] finish/abort/preempt/shutdown 后所有 group 无 block/tensor 泄漏。
- [ ] prefix cache 在支持和不支持的 layout 下都有明确、已测语义。
- [ ] CPU reference、GPU eager、非 model 回归和完整默认回归通过。
- [ ] V1 requirements 和架构文档与最终实现一致。

## 17. 实现状态

2026-07-22 完成技术抽象和执行顺序规划，尚未开始实现。
