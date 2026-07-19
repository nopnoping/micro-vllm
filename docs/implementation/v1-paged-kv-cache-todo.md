# V1.2 Paged KV Cache 实现 TODO

## 1. 目标

在现有统一 token-budget control loop 上加入纯 CPU 的逻辑 Paged KV Cache 管理，为后续 Worker 物理 KV tensor、slot mapping 和 paged attention 提供稳定控制面契约。

实现顺序：

```text
BlockPool + BlockTable
        -> 基础 KVCacheManager
        -> Scheduler block-aware integration
        -> PrefixCache
        -> Preemption + recompute resume
```

本里程碑完成后必须支持：

- 固定数量 physical block ID 的确定性分配；
- request logical block 到 physical block 的映射；
- 按 scheduled token 数原子预留 KV slots；
- finish/abort 后释放 request block；
- 完整 computed block 的 prefix sharing；
- KV 空间不足时抢占并通过 recompute 恢复；
- SchedulerOutput 携带只读 block table snapshot；
- 全部逻辑通过 CPU-only tests，不持有真实 KV tensor。

## 2. 固定设计原则

- [ ] physical block 使用非负整数 ID，不引入可变 block 对象作为公共 API。
- [ ] BlockPool 是 physical block refcount 和 free queue 的唯一所有者。
- [ ] BlockTable 只维护 logical-to-physical mapping，不直接 retain/release block。
- [ ] KVCacheManager 协调 BlockPool、BlockTable、PrefixCache 和 request cache 生命周期。
- [ ] Scheduler 决定调度、finish、abort 和 preempt 哪个 request。
- [ ] KVCacheManager 决定 block 如何分配、共享和释放。
- [ ] Worker/Executor 不参与逻辑 block 分配。
- [ ] Engine/Core 侧不创建 Torch tensor 或 CUDA object。
- [ ] block allocation 在 SchedulerOutput 生成前完成。
- [ ] executor 失败或 update 校验失败时，pending plan 及其 block reservation 可重试。
- [ ] sampled token 在下一 step 计算；本 step 不为刚采样出的 token 分配 slot。
- [ ] 只有成功计算并提交的完整 block 才能进入 PrefixCache。
- [ ] partial block 不共享，不实现 copy-on-write。
- [ ] 本阶段先支持单 KV cache group 和 full attention。

## 3. Cache 配置与错误类型

### `src/micro_vllm/config.py`

- [x] 定义 frozen `CacheConfig`。
- [x] 添加 `block_size: int`。
- [x] 添加 `num_blocks: int`。
- [x] 添加 `enable_prefix_cache: bool`。
- [x] 添加 prefix cache 最大 block 数或明确采用按需 eviction 策略。
- [x] 校验 block size 和 block count 均大于零。
- [x] 暂不添加 dtype、head shape、layer count 和 GPU memory utilization。

### 错误行为

- [x] invalid block 使用 `ValueError`，double free 使用 `RuntimeError`，容量不足返回 `None`。
- [x] insufficient blocks 是可供 Scheduler 处理的资源结果，不使用裸断言。
- [x] internal refcount/free-queue 不变量失败使用明确 `RuntimeError`。
- [x] 错误消息包含 block ID、请求数量和可用容量等适用的诊断信息。

### 测试

- [x] CacheConfig 接受合法配置。
- [x] CacheConfig 拒绝非正 block size/count。
- [x] prefix cache 关闭时相关容量配置行为明确。

## 4. Phase A：BlockPool

### `src/micro_vllm/core/block.py`

#### 初始化

- [x] `BlockPool(num_blocks)` 创建 ID 范围 `[0, num_blocks)`。
- [x] 初始 free queue 顺序确定为 `0..num_blocks-1`。
- [x] 初始所有 block refcount 为零。
- [x] 初始 `num_free_blocks == num_blocks`。
- [x] BlockPool 不保存 Request、token 或 tensor。

#### 分配

- [x] 提供原子 `allocate(count)`。
- [x] `count > 0`。
- [x] 分配成功返回不可变 block ID tuple。
- [x] 新分配 block 的 refcount 设为 1。
- [x] 分配顺序由 free queue 决定，测试可复现。
- [x] 容量不足时不改变 free queue 或任何 refcount。
- [x] 容量不足返回明确结果或抛出专用 `InsufficientBlocksError`，二选一并统一使用。

#### 引用计数

- [x] 提供 `retain(block_id)`。
- [x] 只能 retain 当前已分配的 block。
- [x] retain 后 refcount 加一。
- [x] 提供 `release(block_id)`。
- [x] release 后 refcount 减一。
- [x] refcount 降为零时 block 重新进入 free queue。
- [x] refcount 大于零时不得进入 free queue。
- [x] 检测 invalid block ID、double free 和负 refcount。
- [x] 同一 block 不得在 free queue 中出现两次。

#### 查询与调试

- [x] 提供只读 `num_blocks`。
- [x] 提供只读 `num_free_blocks`。
- [x] 提供 `ref_count(block_id)`。
- [x] 提供只读 free block snapshot，仅用于测试/trace。
- [x] 查询接口不得暴露可修改内部 list/deque。

### `tests/core/test_block.py`：BlockPool

- [x] 初始状态正确。
- [x] 单个和批量分配顺序正确。
- [x] 分配后 refcount 为 1。
- [x] retain/release 状态正确。
- [x] release 到零后 block 可再次分配。
- [x] 部分 release 不释放仍被引用的 block。
- [x] 容量不足的批量分配是原子的。
- [x] count=0/负数被拒绝。
- [x] invalid block ID 被拒绝。
- [x] double free 被拒绝。
- [x] 固定随机种子的 allocate/retain/release 序列始终满足不变量。
- [x] 任意时刻 `allocated + free == num_blocks`，共享引用不重复计算 block 数。

## 5. Phase A：BlockTable

### 数据模型

- [ ] `BlockTable` 构造时接收 `block_size`。
- [ ] logical block index 从零开始。
- [ ] mapping 元素为 `int | None`。
- [ ] `None` 表示该 logical block 当前没有 physical mapping。
- [ ] BlockTable 不验证 physical ID 是否属于某个 BlockPool。
- [ ] BlockTable 不修改 refcount。

### 操作

- [ ] 提供 append 单个或多个 mapping。
- [ ] 提供 `physical_block(logical_block_index)`。
- [ ] 提供 token position 到 `(physical_block_id, offset)` 的 slot 查询。
- [ ] token slot 使用 `divmod(token_position, block_size)` 作为唯一映射规则。
- [ ] 查询 unmapped logical block 时给出明确错误或 `None` 结果，语义统一。
- [ ] 提供按有效 token 长度 `truncate(num_tokens)`。
- [ ] truncate 只移除不再覆盖任何保留 token 的尾部 logical blocks。
- [ ] truncate 返回被移除的非空 physical block IDs，由 KVCacheManager release。
- [ ] partial last block 保持映射，不因 token 数未对齐而释放。
- [ ] 提供 `clear()` 并返回所有非空 physical block IDs。
- [ ] 提供不可变 `snapshot()`。
- [ ] 提供 `num_logical_blocks` 和 `capacity_tokens` 等只读属性。

### `tests/core/test_block.py`：BlockTable

- [ ] 空表状态正确。
- [ ] append 后 mapping 顺序正确。
- [ ] snapshot 不随外部输入或后续修改而变化。
- [ ] 多个 block size 下 token slot 映射正确。
- [ ] block 边界前后 position 映射正确。
- [ ] negative/out-of-range position 被拒绝。
- [ ] `None` mapping 行为正确。
- [ ] truncate 到 block 边界正确释放尾部 block。
- [ ] truncate 到 partial block 保留最后一个 block。
- [ ] truncate 到零等价于清空。
- [ ] clear 返回所有非空 block ID，且 table 为空。
- [ ] BlockTable 操作不会直接改变 BlockPool refcount。

## 6. Phase A 验收

- [x] `core/block.py` 不导入 Request、Scheduler、Executor 或 Torch。
- [ ] BlockPool 和 BlockTable 可独立使用。
- [ ] 所有 block primitive tests 为 CPU-only。
- [x] deterministic randomized invariant test 通过。
- [ ] BlockPool/BlockTable 的所有权和职责 docstring 完整。

## 7. Phase B：基础 KVCacheManager

### `src/micro_vllm/core/kv_cache_manager.py`

#### 状态

- [ ] KVCacheManager 拥有一个 BlockPool。
- [ ] KVCacheManager 按 request ID 拥有 BlockTable。
- [ ] 首版只创建一个 full-attention cache group。
- [ ] 重复注册同一 request ID 被拒绝。
- [ ] 未注册 request 的查询/释放行为明确。

#### Slot 需求计算

- [ ] 根据 `num_computed_tokens + num_scheduled_tokens` 计算 target token capacity。
- [ ] 使用统一 ceil-div 计算 required logical blocks。
- [ ] 已有 partial block 可继续承载 token 时不分配新 block。
- [ ] 跨 block boundary 时只分配缺少的 block 数。
- [ ] 不为刚采样但尚未 scheduled 的 token 提前分配。

#### `allocate_slots()`

- [ ] 接受 request ID、当前 computed token 数和本轮 scheduled token 数，或接受只读 Request。
- [ ] 不修改 Request。
- [ ] 没有新增 block 需求时返回当前 snapshot。
- [ ] 新 block 分配是原子的。
- [ ] 分配成功后 append 到对应 BlockTable。
- [ ] 返回 frozen allocation result。
- [ ] allocation result 至少包含完整 block table snapshot 和 new block IDs。
- [ ] 容量不足时 request table 和 BlockPool 均保持不变。

#### 释放

- [ ] `free_request(request_id)` 清空对应 BlockTable。
- [ ] 对每个非空 physical block release 一次 request 引用。
- [ ] 删除 request 到 BlockTable 的映射。
- [ ] 重复 free 不得产生 double release；选择幂等或明确报错并测试。
- [ ] 提供只读 `get_block_table(request_id)`。

### Frozen result contract

- [ ] 定义 `KVCacheAllocation` 或等价 frozen dataclass。
- [ ] 包含 block table snapshot。
- [ ] 包含本轮 new block IDs。
- [ ] 不包含可变 BlockTable 或 BlockPool 引用。
- [ ] 不包含 Request 或 tensor。

### `tests/core/test_kv_cache_manager.py`

- [ ] 首次分配不足一个 block 的 token。
- [ ] 在 partial block 内继续分配不新增 physical block。
- [ ] 跨一个和多个 block boundary 分配正确。
- [ ] 多 request 的 block table 相互隔离。
- [ ] 多 request 消耗同一 pool 容量。
- [ ] 容量不足时 allocation 原子失败。
- [ ] free request 回收全部 request-owned blocks。
- [ ] 释放一个 request 不影响其他 request。
- [ ] allocation snapshot 不可变。
- [ ] request finish/abort 所需的 free 操作可重复验证。

## 8. Phase C：Scheduler 集成

### 构造与职责

- [ ] Scheduler 显式接收 KVCacheManager。
- [ ] 更新测试 fixture，为每个 Scheduler 提供独立 cache manager。
- [ ] 不提供隐藏的无限容量或 no-op cache fallback。
- [ ] Scheduler 继续拥有 Request 状态；KVCacheManager 不改变 Request。

### 扩展 `ScheduledRequest`

- [ ] 添加只读 block table snapshot。
- [ ] 添加本轮 new block IDs，仅在 Executor/trace 确实需要时保留。
- [ ] 保持 dataclass frozen。
- [ ] block table 中 physical ID 为非负整数或 `None`。
- [ ] 不加入 BlockPool、BlockTable 或 KVCacheManager 可变引用。

### `schedule()` 集成

- [ ] 对候选 request 先计算 token-budget allocation。
- [ ] 调用 KVCacheManager 原子预留 slots。
- [ ] 预留成功后才将 request 放入 SchedulerOutput。
- [ ] waiting request 只有预留成功才进入 RUNNING。
- [ ] pending plan 重复 `schedule()` 不重复分配 block。
- [ ] 本阶段容量不足时停止或跳过 admission，行为确定且有测试。
- [ ] 不在 Scheduler 中复制 block 数计算逻辑。
- [ ] plan 中 block table 能覆盖本轮所有 scheduled token positions。

### `update()` 集成

- [ ] result validation 在 Request/KV cache 提交前完成。
- [ ] 成功 update 后标记对应 tokens 已经计算。
- [ ] Request finish 后调用 `free_request()`。
- [ ] finish 释放发生在所有需要的 cache commit 之后。
- [ ] update 校验失败时保留 pending plan 和已预留 block，允许重试。
- [ ] plan 成功提交后清除 pending reservation 状态。

### abort 集成

- [ ] abort waiting 且从未分配 block 的 request 不影响 pool。
- [ ] abort running request 释放其所有 block。
- [ ] abort pending plan 中 request 时安全取消 plan 或重建 plan。
- [ ] abort 不得泄漏其他 plan request 的 block。
- [ ] abort finished request 不重复释放。

### 测试更新

- [ ] 现有 scheduler/control-loop tests 继续通过。
- [ ] 每个 plan 的 block table 覆盖对应 scheduled token。
- [ ] prompt chunk 跨 block boundary 时逐步追加 block。
- [ ] sampled token 只在下一 step 获得 slot。
- [ ] finish 后 pool 恢复预期 free block 数。
- [ ] abort waiting/running 后 pool 状态正确。
- [ ] invalid executor output 重试不重复分配。
- [ ] equal-but-foreign plan 不释放或污染 block。
- [ ] cache 容量不足时 waiting request 保持 WAITING。

## 9. Phase C 验收

- [ ] V1 需求“finish 释放非共享 KV block”完成。
- [ ] V1 需求“abort/cancel 安全释放资源”完成。
- [ ] SchedulerOutput 包含控制信息和 block IDs，不包含 tensor。
- [ ] BlockPool 无泄漏、double free 或重复 allocation。
- [ ] control-loop FakeExecutor 不需要理解 block allocation。

## 10. Phase D：PrefixCache 数据模型

### Cache key 与 hash

- [ ] 使用确定性 cryptographic hash，例如 `hashlib.blake2b`。
- [ ] 不使用进程随机化的 Python `hash()` 作为持久 cache identity。
- [ ] block hash 包含 parent block hash。
- [ ] block hash 包含当前完整 token block。
- [ ] block hash 包含 cache namespace/model identity。
- [ ] 为未来 cache group ID 留出明确字段，但首版固定单 group。
- [ ] 相同 token block 但不同 parent prefix 不得误共享。
- [ ] 相同 prefix 和 namespace 生成稳定相同 hash。

### Prefix cache entry

- [ ] entry 至少包含 block hash、physical block ID 和 token block identity。
- [ ] PrefixCache 持有每个 cached block 的一份 cache pin reference。
- [ ] active request 使用 cached block 时额外 retain request reference。
- [ ] request 结束只释放 request reference，不释放 cache pin。
- [ ] eviction 删除 cache entry 并释放 cache pin。
- [ ] 若 block 仍被 active request 使用，eviction 后 physical block 继续存活到 request release。
- [ ] PrefixCache 使用确定性 LRU 或 FIFO eviction，并文档化选择。
- [ ] cache capacity 有明确上限，不允许无限 pin 导致 pool 永久耗尽。

### Cache pressure

- [ ] BlockPool 容量不足前，KVCacheManager 可以请求 PrefixCache eviction。
- [ ] eviction 优先回收只有 cache pin 的 block。
- [ ] eviction 不修改 active request block table。
- [ ] eviction 后相应 hash lookup 失效。
- [ ] eviction 和分配组合保持原子或具备安全 rollback。

## 11. Phase D：Prefix lookup 与 commit

### 查找

- [ ] `find_prefix_hit()` 只按完整 token block 顺序查找。
- [ ] 命中必须从第一个 logical block 连续开始。
- [ ] 遇到第一个 miss 后停止，不跳过中间 block。
- [ ] 命中 block 为 request 增加 refcount。
- [ ] 命中 block append 到 request BlockTable。
- [ ] lookup 失败或中途异常时回滚本次已 retain blocks。
- [ ] 返回命中 block IDs 和命中 token 数，不直接修改 Request。

### 保留采样所需 token

- [ ] prefix hit 后至少保留一个已知 token 由模型实际计算，以获得 logits。
- [ ] prompt 长度不是 block size 整数倍时，只复用完整前缀 blocks。
- [ ] prompt 恰好按 block 对齐时，不复用最后一个完整 block，避免无 token 可计算。
- [ ] 首版不对共享的最后 block 执行 copy-on-write。
- [ ] 上述规则有 block-aligned prompt 专项测试。

### 计算提交

- [ ] 只有 Scheduler.update 成功后调用 prefix commit。
- [ ] 只提交 `num_computed_tokens // block_size` 范围内完整 computed blocks。
- [ ] partial block 不进入 cache。
- [ ] 同一 request/block 不重复插入或重复 pin。
- [ ] 相同 block hash 已存在时复用 canonical cached block 的策略明确。
- [ ] 若 newly computed block 与已缓存 canonical block 重复，处理替换/释放而不泄漏引用。
- [ ] request finish 前先完成可缓存 block commit，再释放 request references。

### Prefix tests

- [ ] 相同完整 prefix 命中。
- [ ] 不同 token、parent hash 或 namespace 不命中。
- [ ] partial block 不命中。
- [ ] 中间 block miss 后不继续命中。
- [ ] cache hit 增加 request refcount。
- [ ] request finish 后 cache pin 保留 block。
- [ ] eviction 释放 cache pin。
- [ ] active shared block 被 eviction 后仍可由 active request 使用。
- [ ] capacity pressure 触发确定性 eviction。
- [ ] block-aligned prompt 至少重算最后一个 block。
- [ ] prefix cache on/off 最终 token 输出一致。

## 12. Phase E：Preemption 与 recompute

### 抢占策略边界

- [ ] Scheduler 决定 victim，KVCacheManager 只执行 victim cache release。
- [ ] 默认 victim policy 独立封装，避免散落在 schedule 主循环。
- [ ] 首版采用确定性策略，例如最近进入的、尚未加入当前 plan 的 running request。
- [ ] 不抢占已执行但尚未 update 的 pending plan request。
- [ ] waiting admission 不抢占高优先级 running decode；行为明确并测试。

### Request 状态

- [ ] RUNNING request 可转换到 PREEMPTED。
- [ ] preempt 后释放 request BlockTable 的全部 request references。
- [ ] preempt 后重置 `num_computed_tokens`，但保留 prompt/output token IDs。
- [ ] Request 提供受控的 recompute reset/resume 操作，不直接修改字段。
- [ ] PREEMPTED request 被放入明确的 resume queue。
- [ ] resumed request 优先级相对新 WAITING request 有明确策略。
- [ ] resume 时可重新执行 prefix lookup。
- [ ] prefix hit token 数由 Scheduler 提交到 Request 进度。
- [ ] 无 prefix hit 时从零 recompute。

### Scheduler 行为

- [ ] allocation 失败时只选择足够释放容量的 victim 数量。
- [ ] victim 选择和释放前先确认计划可行，避免无意义抢占。
- [ ] 抢占成功后重试当前 request allocation。
- [ ] 无可抢占 victim 时当前 request 保持可恢复状态，不产生空转死循环。
- [ ] request 最终能够在资源可用后完成，避免永久 starvation。
- [ ] preemption 事件写入最小可测试 decision record 或状态输出。

### Preemption tests

- [ ] 单 request cache 不足时行为明确且不死循环。
- [ ] 一个 request 抢占另一个 request 后获得 blocks。
- [ ] victim block 全部释放，无 refcount 泄漏。
- [ ] victim tokens 保留但 computed progress 被重置。
- [ ] resume 无 prefix hit 时从零 recompute。
- [ ] resume 有 prefix hit 时从完整 cached blocks 后继续。
- [ ] preempt/resume 最终输出与无抢占执行一致。
- [ ] 多次 preempt/resume 不 double free。
- [ ] abort PREEMPTED request 不泄漏 cache state。
- [ ] prefix cache pin 不被 request preemption 错误释放。
- [ ] running order、resume priority 和 victim 选择可复现。

## 13. 可观察性

- [ ] BlockPool 提供 free/allocated/refcount 只读 snapshot。
- [ ] KVCacheManager 可以报告 per-request block table。
- [ ] 记录或可测试获取 allocate/free/retain/release 事件。
- [ ] 记录 prefix hit block/token 数。
- [ ] 记录 prefix insert/evict。
- [ ] 记录 preempt/resume 和 recompute token 数。
- [ ] 调试数据不暴露可修改内部集合。
- [ ] 关闭详细事件记录时不改变 cache 语义。

## 14. 文档与需求同步

- [ ] 为 BlockPool、BlockTable、KVCacheManager 和 PrefixCache 记录所有权。
- [ ] 文档说明 request ref 与 cache pin ref 的区别。
- [ ] 文档说明 partial block 不共享。
- [ ] 文档说明 block-aligned prompt 为什么需要保留模型计算。
- [ ] 文档说明 preemption 使用 recompute、不使用 CPU swap。
- [ ] 只勾选经过测试验证的 V1 requirement 条目。
- [ ] 更新 control-loop 文档中受 block-aware Scheduler 影响的接口描述。

## 15. 推荐提交顺序

1. [ ] `feat: add deterministic block pool and block table`
2. [ ] `feat: add basic logical KV cache manager`
3. [ ] `feat: integrate KV block allocation with scheduler`
4. [ ] `feat: add prefix block hashing and cache sharing`
5. [ ] `feat: add cache-aware preemption and recompute resume`
6. [ ] `docs: mark completed paged KV cache requirements`

每个提交应保持 CPU tests 可运行，避免将 BlockPool、PrefixCache 和 preemption 作为一个不可独立 review 的大提交。

## 16. 明确不实现

- [ ] 不创建实际 K/V tensor。
- [ ] 不实现 Worker KV cache initialization。
- [ ] 不实现 slot mapping tensor。
- [ ] 不实现 attention kernel。
- [ ] 不实现 sliding window block recycling。
- [ ] 不实现 chunked-local block recycling。
- [ ] 不实现多 KV cache group。
- [ ] 不实现 CPU swap/offload。
- [ ] 不实现 remote KV cache 或 connector。
- [ ] 不实现 partial block sharing/copy-on-write。
- [ ] 不实现跨进程 block synchronization。
- [ ] 不实现基于 GPU memory profiling 的自动 block count。

## 17. 最终验收

- [ ] BlockPool、BlockTable 和 KVCacheManager 全部通过 CPU-only tests。
- [ ] Scheduler 任意 step 不分配超过 pool capacity 的 block。
- [ ] plan block table 覆盖所有 scheduled token positions。
- [ ] finish、abort、preempt 后没有 request-owned block 泄漏。
- [ ] prefix sharing 正确增加/释放 request 与 cache pin references。
- [ ] cache pressure 能确定性 eviction，不永久 pin 满整个 pool。
- [ ] prefix cache on/off 输出一致。
- [ ] preempt/resume 与无抢占输出一致。
- [ ] 无 double free、负 refcount、重复 free queue entry。
- [ ] 现有 control-loop tests 继续通过。
- [ ] 运行全部 unit/integration tests。
- [ ] 运行 compile、lint 和 type check；未配置项目明确记录。
- [ ] 检查源码不导入 Torch/CUDA。
- [ ] 检查工作区只包含 V1.2 相关改动。
