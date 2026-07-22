# V1.5 Multi-Attention KV Cache 实现清单

## 1. 目标

在不改变 Request 进度模型、token-budget Scheduler、victim selection 和 Engine
control loop 的前提下，让同一个模型中的 full attention、sliding-window
attention 和 chunked-local attention 使用各自的逻辑 cache 规则，同时共享一个
全局 BlockPool 和一套物理 KV page。

最终数据流固定为：

```text
model per-layer Attention
    -> ModelRunner derives stable cache groups and ordered layer indices
    -> PhysicalKVCache allocates max_group_layers shared buffers
    -> KVCacheManager coordinates one group manager per cache group
    -> all groups allocate distinct block IDs from one BlockPool
    -> Scheduler sees only common cached-prefix length and allocation success
    -> ModelRunner builds one AttentionMetadata per group
    -> each model layer receives its bound physical buffer and group metadata
```

V1.5 必须支持：

- full causal attention；
- sliding-window attention；
- chunk-aligned local attention；
- full + sliding、full + chunked-local hybrid model；
- group-aware prefix cache；
- bounded group 在成功 step 后释放不可见 block；
- 不同 group 的相同 `group_layer_index` 共享物理 buffer 地址；
- 所有 group 从一个全局 BlockPool 分配互斥 block ID。

## 2. 范围和非目标

### 本阶段实现

- ModelRunner 从 model 的 per-layer Attention 声明派生 cache topology。
- 相同 `AttentionSpec` 的 layer 进入同一个 cache group。
- 每 request/group 拥有独立 absolute-index BlockTable。
- 所有 group 共享一个 BlockPool、prefix-cache LRU 和 null block。
- Full、SlidingWindow、ChunkedLocal 分别实现 cache-hit、allocation 和 release 规则。
- SchedulerOutput 和 ModelInput 统一携带 per-group 数据。
- PhysicalKVCache 按最大 group layer 数分配共享 buffers。
- synthetic hybrid model 和 Mistral-family 模型端到端验收。

### 本阶段不实现

- optimized/CUDA attention operator；继续使用 reference operator。
- 不同 group 使用不同 token block size、dtype 或独立 BlockPool。
- cyclic/ring BlockTable。
- 在同一个 model step 内回收并复用刚刚越出 window/chunk 的 block。
- CPU KV swap/offload、KV connector 或 disaggregated serving。
- 根据 group layer 数自动拆分过长 group；首版允许较短 group 的 page padding。
- 通用声明式 Transformer builder。
- 修改 Scheduler 的 token budget、FIFO、victim selection 或 recompute 策略。

## 3. 核心 contract

### 3.1 AttentionSpec

`AttentionSpec` 是不含 tensor/device 的 frozen semantic value。对 absolute query
position `p`，可见 key 区间统一定义为：

```text
context_start(p) <= key_position <= p
```

| Spec | `context_start(p)` |
|---|---|
| `FullAttentionSpec` | `0` |
| `SlidingWindowSpec(window_size)` | `max(0, p - window_size + 1)` |
| `ChunkedLocalSpec(chunk_size)` | `(p // chunk_size) * chunk_size` |

Attention visibility、prefix-hit 和 release 是三个相关但不同的概念：

- `context_start()` 只定义 operator 可见性；
- prefix-hit 由对应的 cache group manager 实现；
- skipped/release boundary 也由对应的 cache group manager 实现。

不把 prefix 或回收规则塞入 Scheduler，也不通过 operator name 推断 cache 类型。

### 3.2 Model 声明和 cache topology

Model 不创建 KVCacheGroup，也不导入 `core`。Model 只按 layer 顺序暴露：

```python
layer_attentions: tuple[Attention, ...]
```

ModelRunner 从 `attention.spec` 派生 cache topology。相同 spec value 的 layer
进入同一个 group；不同 window/chunk 参数属于不同 group。group ID 按 spec 在
model layer 顺序中第一次出现的顺序，从 `0` 连续编号。

```python
@dataclass(frozen=True, slots=True)
class KVCacheGroupSpec:
    group_id: int
    attention_spec: AttentionSpec
    layer_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KVCacheTopology:
    groups: tuple[KVCacheGroupSpec, ...]
    page_layer_count: int
```

`layer_indices` 是唯一的 layer/group 映射来源。layer 在所属 group 的
`layer_indices` tuple 中的位置就是 `group_layer_index`，不重复保存反向 binding。
`page_layer_count` 为：

```text
max(len(group.layer_indices) for group in groups)
```

ModelRunner 在 memory profiling 得到 capacity 后，将 topology 和物理参数冻结为：

```python
@dataclass(frozen=True, slots=True)
class KVCacheLayout:
    block_size: int
    num_blocks: int
    topology: KVCacheTopology
```

内部 topology/layout 由 typed model declaration 和稳定构造流程产生，不重复做
运行时校验。window/chunk 等外部模型配置值仍在其输入边界做 domain validation。

### 3.3 物理 page 和全局 block namespace

假设有两个 group：

```text
full = [full.0, full.1]
sw   = [sw.0, sw.1, sw.2, sw.3]
```

则 `page_layer_count = 4`，PhysicalKVCache 只分配四个 buffers：

```text
buffer[0] <- full.0 / sw.0
buffer[1] <- full.1 / sw.1
buffer[2] <-          sw.2
buffer[3] <-          sw.3
```

不同 group 共享 buffer 地址，但不能同时使用同一个 block ID。若 full 使用 block
7，它写入 `buffer[0][7]` 和 `buffer[1][7]`；sw 必须使用另一个 block ID。

因此：

```text
bytes_per_block =
    page_layer_count
    * 2
    * block_size
    * num_kv_heads
    * head_dim
    * element_size
```

`num_blocks` 表示物理 page 总数。所有 group 的 BlockTable 都引用同一个
`[0, num_blocks)` namespace。

### 3.4 公共 null block

BlockPool 初始化时永久保留一个公共 null block：

- null block 使用最后一个真实 block ID，即 `num_blocks - 1`；
- 不进入 free queue；
- 不参与普通 refcount、hash、eviction 或重新分配；
- 所有 group 已跳过的 logical block 都替换为该 ID；
- rectangular block-table padding 也使用该 ID；
- `num_free_blocks` 只统计可分配 block，不包含 null block。

因此 `num_blocks` 至少为 `2`，并始终表示包含 null page 在内的物理 page 总数。

BlockTable 因此统一保存 `int`，不再用 `None` 或负数表示 hole。attention mask
保证 null block 对应的不可见位置不影响输出。

### 3.5 Prefix cache identity 和公共前缀

Prefix block hash 的 domain 显式包含 group ID：

```text
hash(group_id, cache_namespace, parent_block_hash, token_block)
```

`group_id` 不属于 Scheduler-facing API。KVCacheManager 根据 topology 创建每个
group manager 时绑定对应 group ID，group manager 只在内部计算 prefix identity。
Scheduler、Request 和 allocation 调用方均不传入或读取 group ID。

相同 token prefix 在不同 group 中分别缓存和淘汰。一个物理 block 在任一时刻只
属于一个 group/cache identity。

每种 group manager 实现：

```python
find_longest_cache_hit(token_ids, max_length)
get_new_block_indices(block_table, num_computed_tokens, num_scheduled_tokens)
cache_computed_blocks(...)
remove_skipped_blocks(...)
```

这些都是 KVCacheManager 内部接口。group manager 持有 topology 绑定的 group ID，
引用公共 BlockPool，但不创建或拥有独立 capacity。hit snapshot 只保存公共 token
boundary 和一份 absolute cached/null block table，不重复保存 cached block ID 列表。

各类型 hit 规则：

- Full：从左到右扫描 chained block hash，第一个 miss 后停止。
- SlidingWindow：对候选 prefix boundary，从右向左寻找足够覆盖下一 query
  window 的连续 cached blocks，window 之前填 null block。
- ChunkedLocal：候选 boundary 之前完成的 chunk 填 null block，只从当前 chunk
  起点向右检查连续 cached blocks。

公共 cached prefix 不能只计算一次后直接取最小值。SW/chunked 所需 blocks 会随
候选 boundary 改变，必须单调收敛：

```text
candidate = max_cache_hit_length
loop:
    每个 group 在 candidate 上重新计算最长有效 hit
    next_candidate = min(group_hit_lengths)
    next_candidate == candidate 时完成
    否则 candidate = next_candidate
```

最终结果同时包含一个公共 `num_computed_tokens` 和每个 group 在该 boundary 下
有效的 cached/null block table。

### 3.6 Scheduler-facing API

Scheduler 不读取 AttentionSpec，不循环 cache groups，也不计算 release boundary。
它只消费：

```python
cache_hit = kv_cache_manager.find_longest_cache_hit(...)
cache_hit.num_computed_tokens

allocation = kv_cache_manager.allocate_slots(
    request_id,
    num_computed_tokens=cache_hit.num_computed_tokens,
    num_scheduled_tokens=num_scheduled_tokens,
    cache_hit=cache_hit,
)
```

运行中的 request 不需要 cache hit，直接把当前公共 `num_computed_tokens` 交给
同一 allocation facade。cache namespace 在 KVCacheManager 构造时绑定，不由
Scheduler 在 lookup 或 commit 时传入。

分配流程由 KVCacheManager 完成：

1. 在最终公共 prefix 上恢复每个 group 的 cached/null blocks。
2. 计算 scheduled token range 对每个 group 的新 block 需求。
3. 汇总 cached acquire 和 new allocation，对共享 BlockPool 做一次 preflight。
4. 只有总需求可容纳时，才按稳定 group 顺序提交全部修改。
5. 失败时不改变 free queue、refcount、cache index 或任何 BlockTable。

Scheduler 仍负责 token/sequence budget、request 状态和 victim selection。

### 3.7 Step commit 和 block release

只有一次 model execution 成功并进入 `Scheduler.update()` 后，才调用：

```python
kv_cache_manager.commit_step(
    request_id,
    token_ids,
    num_computed_tokens,
)
```

KVCacheManager 对每个 group：

1. 将新完成且可缓存的 blocks 写入 group-aware prefix index。
2. 计算已不再需要的完整 logical blocks。
3. 将这些 table entries 替换为公共 null block。
4. 将原 physical blocks 释放到公共 pool/LRU。

skipped-token 基础规则：

| Spec | skipped prefix tokens |
|---|---|
| Full | `0` |
| SlidingWindow | `max(0, num_computed_tokens - window_size + 1)` |
| ChunkedLocal | `(num_computed_tokens // chunk_size) * chunk_size` |

只释放完整落在 skipped prefix 内的 block；visibility boundary 所在的 partial
block 保留。plan retry、failed execution 或尚未 update 时不得提前释放。

### 3.8 ModelRunner execute

ModelRunner 对一个 batch：

1. 构造共享 input IDs、positions、query row ordering 和 sampling rows。
2. 每个 group 分别构造 block table tensor 和 slot mapping。
3. 根据 group spec 构造一个 AttentionMetadata。
4. 枚举每个 group 的有序 `layer_indices`，展开为 model layer 顺序：

```python
for group in topology.groups:
    for group_layer_index, layer_index in enumerate(group.layer_indices):
        layer_cache[layer_index] = physical_buffers[group_layer_index]
        layer_metadata[layer_index] = group_metadata[group.group_id]
```

Model forward 只按 layer 顺序消费 cache 和 metadata，不查询全局 cache registry。

## 4. Phase 0：冻结 model/cache topology contract

- [x] Model 统一暴露 `layer_attentions`，不再声明 cache group。
- [x] `KVCacheGroupSpec.layer_indices` 作为唯一 layer/group 映射来源。
- [x] 新增 frozen `KVCacheTopology`。
- [x] `KVCacheLayout` 显式组合 topology 和初始化后的 block capacity。
- [x] 实现从 per-layer AttentionSpec 派生 topology 的纯函数。
- [x] 相同 spec value 合并，group ID 按首次出现顺序稳定生成。
- [x] 每个 group 内 `group_layer_index` 从 `0` 连续编号。
- [x] `page_layer_count` 等于最大 group layer 数。
- [x] full-only Llama/Qwen 自然退化为一个 group，不保留特殊 topology。
- [x] contract dataclass 只依赖 typed internal input，不增加重复 runtime validation。
- [x] 单元测试覆盖 full-only、hybrid、不同 window 参数和不等长 group。

Phase 0 不修改 BlockPool、KVCacheManager、SchedulerOutput、PhysicalKVCache 或实际
execute 数据流。

## 5. Phase 1：公共 null block 和 group-aware BlockPool

- [x] BlockPool 保留一个不进入 free queue 的 null block。
- [x] BlockTable 使用 null block ID 表示 skipped/hole，不再保存 `None`。
- [x] rectangular padding 使用 null block ID，不再使用负 block ID。
- [x] prefix block hash domain 包含 `group_id`。
- [x] cached block reuse/eviction 保留 deterministic free ordering。
- [x] capacity preflight 同时计算 cached acquire 和 new blocks。
- [x] null block 不改变 refcount，不可被 free、cache 或 evict。
- [x] invariant test 覆盖 null、active、cached-free 和 reused block 生命周期。

## 6. Phase 2：per-group cache policy

- [x] 定义小型 `CacheGroupManager` interface，不把类型分支堆进 Scheduler。
- [x] group manager 构造时绑定 group ID，不由 Scheduler/API 调用方传入。
- [x] 实现 `FullCacheGroupManager`。
- [x] 实现 `SlidingWindowCacheGroupManager`。
- [x] 实现 `ChunkedLocalCacheGroupManager`。
- [x] Full prefix 从左到右遇 miss 停止。
- [x] SW prefix 从候选 boundary 向左验证连续 window blocks。
- [x] Chunked prefix 将完成 chunk 视为 null，只验证当前 chunk。
- [x] manager 返回 scheduled range 缺失的 absolute logical block indices。
- [x] 三种 manager 分别实现 skipped-token/release rule。
- [x] cache hit 返回当前候选 boundary 对应的完整 cached/null snapshot。
- [x] 不同 group manager 共享同一个 BlockPool，不拥有独立 capacity。

## 7. Phase 3：KVCacheManager coordinator 和 Scheduler facade

- [x] KVCacheManager 根据 layout 创建一个 group manager per group。
- [x] 实现跨 group fixed-point common-prefix intersection。
- [x] `KVCacheHit` 同时携带公共 token 数和 per-group opaque hit state。
- [x] 每 request/group 拥有独立 BlockTable。
- [x] allocation 汇总所有 group 需求并原子提交共享 pool。
- [x] Scheduler 不读取 group/spec，只读取公共 hit length 和 allocation 成败。
- [x] cache namespace 在 KVCacheManager 构造时绑定，不经过 Scheduler。
- [x] `commit_step()` 统一完成 prefix commit 和 skipped block release。
- [x] plan retry 和 failed execution 不改变已提交 cache state。
- [x] preempt/recompute、abort 和 finish 一次性释放所有 group。
- [x] victim selection 保持现有 deterministic Scheduler 策略。

Phase 3 先由 coordinator 生成 per-group tables；Phase 5 已将
SchedulerOutput/ModelRunner 数据 contract 统一为有序 group snapshots。

## 8. Phase 4：PhysicalKVCache 共享 page layout

- [x] memory profiling 前由 ModelRunner 派生 KVCacheTopology。
- [x] bytes-per-block 使用 `page_layer_count`，不使用 model 总 layer 数。
- [x] PhysicalKVCache 只创建 `page_layer_count` 个 buffers。
- [x] 所有 buffer 共享相同全局 block dimension。
- [x] group layer 通过 `group_layer_index` 绑定到共享 buffer。
- [x] 较短 group 的未使用 page slices 允许 padding 浪费。
- [x] warmup/profile 使用与正式执行相同的 topology 和 page size。
- [x] Worker 只读暴露初始化后的 KVCacheLayout。
- [x] LLM 使用 Worker layout 构造逻辑 KVCacheManager。
- [x] CPU deterministic capacity 和 CUDA profiled capacity 使用相同含义。

## 9. Phase 5：per-group SchedulerOutput 和 ModelRunner execute

- [x] 引入 frozen `CacheGroupBlockTable`。
- [x] `ScheduledRequest` 统一携带 group snapshots，移除单 `block_table` 特例。
- [x] 每 group 构造独立 padded block-table tensor。
- [x] 每 group 构造独立 slot mapping。
- [x] 每 group/spec 构造独立 AttentionMetadata。
- [x] common positions/query row ordering 只构造一次。
- [x] 按 group 的有序 layer indices 展开 physical buffer 和 metadata。
- [x] sampling row、request ID 和 logits selection 不受 group 数量影响。
- [x] SchedulerOutput 不含 tensor、Request、model object 或 mutable container。
- [x] full-only 模型也走一元 group tuple 的统一路径。

## 10. Phase 6：hybrid correctness 和生命周期验收

- [ ] 构造 test-only full + chunked-local tiny model。
- [ ] 构造 test-only full + sliding-window tiny model。
- [ ] 一次 full prefill 与多次 chunked prefill 的最终 hidden/logits 一致。
- [ ] prefill + decode 与显式 dense per-layer mask reference 对拍。
- [ ] mixed request batch 与逐 request 独立执行一致。
- [ ] common-prefix fixed point 覆盖 candidate 下降后 SW/chunk hit 改变的场景。
- [ ] local group 释放旧 block 时 full group 保留完整历史。
- [ ] released table entries 全部指向同一个 null block。
- [ ] reused physical block 不产生 stale KV 污染。
- [ ] 任一 group 容量失败不会留下部分 reservation。
- [ ] finish/abort/preempt 后所有真实 block refcount 归零。

## 11. Phase 7：Mistral-family 集成

- [ ] 完成 `models/mistral.py`，不在 Loader/Runner/Scheduler 添加 Mistral 分支。
- [ ] 从 ModelConfig 读取结构参数和 sliding-window size。
- [ ] 每个 decoder layer 暴露正确的 SlidingWindowSpec。
- [ ] 复用已有 RMSNorm、RoPE、SwiGLU、Embedding/LMHead 和 Linear。
- [ ] 实现 checkpoint name mapping、tied weights 和 registry 映射。
- [ ] 使用 tiny Transformers Mistral checkpoint 对拍参数、hidden 和 logits。
- [ ] 验证 full prefill、chunked prefill、decode、window 回收和 reuse。
- [ ] 选定固定公开 checkpoint，记录 repo、revision、dtype 和 T4 memory budget。
- [ ] 公开 checkpoint 支持标准 Hub cache 离线重载。

## 12. Phase 8：回归、可观察性和文档

- [ ] 现有 Llama/Qwen full-only 数值结果不变。
- [ ] 现有 prefix cache、preemption、block reuse 和 loaded-model 对拍不回归。
- [ ] structured trace 按 group 报告 spec、layers、block table 和 release 数量。
- [ ] trace 报告 page_layer_count、physical buffers、dtype 和总字节。
- [ ] 更新 `docs/requirements/v1-core-engine.md`。
- [ ] 更新 requirements README 的稳定 contract。
- [ ] 记录 full、SW、chunked 和 hybrid 的最小数据流示例。
- [ ] 完成非 model、tiny model、GPU eager 和固定公开模型回归。

## 13. 测试分层

| 层级 | 目录/文件 | 重点 |
|---|---|---|
| Topology | `tests/core/test_kv_cache_spec.py` | stable grouping、bindings、page width |
| Attention | `tests/attention/operators/test_reference.py` | visible keys 与 dense mask |
| Block pool | `tests/core/test_block.py` | null block、group identity、LRU/refcount |
| Group policy | `tests/core/test_cache_group_manager.py` | Full/SW/Chunk hit 和 release |
| Coordinator | `tests/core/test_kv_cache_manager.py` | fixed point、atomic allocation、lifecycle |
| Scheduler | `tests/core/test_scheduler.py` | opaque hit、capacity failure、preemption |
| Physical cache | `tests/worker/test_kv_cache.py` | page width、shared addresses、capacity |
| Runner | `tests/worker/test_model_runner_execution.py` | per-group metadata 和 layer binding |
| Hybrid | `tests/integration/test_hybrid_attention_engine.py` | 数值与生命周期端到端 |
| Mistral | `tests/models/test_mistral_model.py` | config、weights、Transformers 对拍 |

测试不得只检查“运行不报错”。至少验证 visible key positions、hidden/logits、
token IDs、block identity 或 refcount/free ordering 中的一项确定性 reference。

## 14. 预计文件变更

```text
src/micro_vllm/attention/spec.py
src/micro_vllm/attention/attention.py
src/micro_vllm/core/kv_cache_spec.py
src/micro_vllm/core/block.py
src/micro_vllm/core/cache_group_manager.py
src/micro_vllm/core/kv_cache_manager.py
src/micro_vllm/core/scheduler_output.py
src/micro_vllm/core/scheduler.py
src/micro_vllm/worker/kv_cache.py
src/micro_vllm/worker/model_runner.py
src/micro_vllm/worker/worker.py
src/micro_vllm/llm.py
src/micro_vllm/models/llama.py
src/micro_vllm/models/qwen.py
src/micro_vllm/models/mistral.py
src/micro_vllm/models/registry.py
```

如果实现需要修改 Engine、Request、Sampler、token budget 或 victim selection
contract，应先停止并重新 review；正常 V1.5 扩展不需要改变这些边界。

## 15. 建议提交顺序

1. `refactor: derive kv cache topology from model layers`
2. `feat: reserve shared null kv cache block`
3. `feat: add attention-specific cache group managers`
4. `feat: coordinate hybrid kv cache groups`
5. `refactor: share physical kv cache pages across groups`
6. `refactor: build attention metadata per cache group`
7. `test: validate hybrid kv cache lifecycle`
8. `feat: add mistral sliding-window model`
9. `docs: complete multi-attention kv cache milestone`

每个提交必须保持可读、可回归的中间状态，不跨越未冻结的 contract。

## 16. 完成标准

- [ ] ModelRunner 是 model layer 声明到 cache topology 的唯一转换边界。
- [ ] Scheduler 不包含 Full/SW/Chunked 分支。
- [ ] 所有 group 共享一个 BlockPool 和一个 null block。
- [ ] group-aware prefix hit 通过 fixed-point 得到真正公共 boundary。
- [ ] 多 group allocation 失败不会留下部分状态。
- [ ] PhysicalKVCache buffer 数等于最大 group layer 数。
- [ ] 不同 group 的相同 group-layer index 使用同一 buffer 地址和不同 block ID。
- [ ] bounded group 可回收旧 block，full group 保留完整历史。
- [ ] full、SW、chunked 和 hybrid 都有 deterministic correctness reference。
- [ ] finish/abort/preempt/shutdown 后无 block/tensor 泄漏。
- [ ] Llama、Qwen、Mistral 和现有 Engine 行为通过回归。

## 17. 实现状态

2026-07-22：完成第二轮技术方案 review、实现清单重写，以及 Phase 0 至
Phase 5。
ModelRunner 从 `layer_attentions` 派生稳定 topology，`groups[].layer_indices` 是
唯一的 layer/group 映射来源。BlockPool 的最后一个物理 page 现在永久作为公共
null block，BlockTable 和 rectangular padding 只保存真实非负 block ID；prefix
hash domain 已包含 group ID。Full、SlidingWindow 和 ChunkedLocal group policy
已使用同一个 BlockPool 实现。KVCacheManager 现在负责 fixed-point common hit、
跨 group 原子 allocation、step commit/release 和完整 request lifecycle；
Scheduler 不再持有 cache namespace 或 cached block ID。按当前要求只完成本地
静态检查。PhysicalKVCache 按 `page_layer_count` 分配共享 buffers，ModelRunner
冻结 layout 并按 group-layer index 展开 layer 地址，Worker 将同一 layout 暴露给
控制面。SchedulerOutput 现在携带 immutable per-group block snapshots，
ModelRunner 为每个 group 构造 block table、slot mapping 和 metadata，再按
`layer_indices` 展开给模型层。未执行 SSH 远端测试。
