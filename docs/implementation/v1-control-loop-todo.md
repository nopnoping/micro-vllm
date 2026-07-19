# V1 首个实现里程碑：Control Loop TODO

## 1. 目标

实现第一条可测试的端到端控制流：

```text
add_request
    -> Scheduler.schedule()
    -> SchedulerOutput
    -> Executor.execute()
    -> Scheduler.update()
    -> Engine output / next step
```

本里程碑使用测试内的 deterministic FakeExecutor，不接入 Torch、CUDA、真实模型和 KV Cache。目标是冻结后续模块共同依赖的请求状态、统一 token 调度和执行计划边界。

## 2. 完成定义

- [ ] 能动态添加 tokenized request。
- [ ] Engine 能重复执行 `schedule -> execute -> update`。
- [ ] 长 prompt 能按照 token budget 跨多个 step 计算。
- [ ] running decode 和 waiting prefill 能进入同一个 execution plan。
- [ ] 只有计算到最后一个已知 token 的请求才产生 sampled token。
- [ ] 请求能因 EOS、max tokens、abort 进入终态。
- [ ] SchedulerOutput 不包含可变 Request，也不包含设备 tensor。
- [ ] 全部逻辑通过 CPU-only unit/integration tests。

## 3. 实现约束

- [ ] 不增加 `is_prefill`、`is_decode` 等持久化 phase 字段。
- [ ] `num_computed_tokens` 是请求计算进度的唯一权威字段。
- [ ] Scheduler 拥有 Request 状态和状态迁移。
- [ ] Executor 只消费执行计划并返回结果。
- [ ] FakeExecutor 只能存在于测试代码中。
- [ ] 本里程碑不创建或模拟 block ID、block table、slot mapping。
- [ ] 数据结构优先使用 typed dataclass；执行计划使用 frozen dataclass。
- [ ] 不为尚未实现的分布式、量化或模型后端增加接口字段。

## 4. Phase 0：工程前置

### `pyproject.toml`

- [x] 确定最低 Python 版本（Python 3.10）。
- [x] 配置 `src/` layout 的 package discovery。
- [x] 添加最小测试依赖 `pytest`。
- [x] 配置 pytest test path。
- [x] 暂不添加 Torch、Transformers、safetensors 和 GPU 依赖。
- [x] 确认 `python -m pytest` 可以发现测试。

### 基础检查

- [x] 确认 `micro_vllm` 可以从开发环境导入。
- [x] 确认现有 skeleton module 均可通过 Python 编译。
- [x] 将确定后的测试命令补充到根 `AGENTS.md`，仅记录稳定命令。

## 5. Phase 1：配置和 SamplingParams

### `src/micro_vllm/config.py`

- [x] 定义最小 `SchedulerConfig`。
- [x] 添加 `max_num_batched_tokens: int`。
- [x] 添加 `max_num_seqs: int`。
- [x] 校验两个值均大于零。
- [x] 暂不引入 GPU、cache、model 或 parallel 配置。

### `src/micro_vllm/sampling/params.py`

- [x] 定义 frozen `SamplingParams`。
- [x] 添加 `max_tokens: int`。
- [x] 添加 `eos_token_id: int | None`。
- [x] 添加 `ignore_eos: bool`。
- [x] 校验 `max_tokens > 0`。
- [x] 暂不添加 temperature、top-k、top-p 和 random seed。

### 测试

- [x] 非法 SchedulerConfig 能给出明确错误。
- [x] 非法 SamplingParams 能给出明确错误。
- [x] frozen SamplingParams 不能被原地修改。

## 6. Phase 2：Request Domain Model

### `src/micro_vllm/core/request.py`

- [ ] 定义 `RequestStatus` enum。
- [ ] 状态至少包含 `WAITING`、`RUNNING`、`PREEMPTED`、`FINISHED`。
- [ ] 定义 `FinishReason` enum。
- [ ] FinishReason 至少包含 `EOS`、`LENGTH`、`ABORTED`。
- [ ] 定义 Request dataclass。
- [ ] 添加稳定 `request_id`。
- [ ] 保存 prompt token IDs。
- [ ] 保存 output token IDs。
- [ ] 保存 `num_computed_tokens`。
- [ ] 保存 SamplingParams。
- [ ] 保存 status 和可选 finish reason。

### Request 计算属性

- [ ] `num_prompt_tokens`。
- [ ] `num_output_tokens`。
- [ ] `num_known_tokens`。
- [ ] `num_pending_tokens`。
- [ ] `all_token_ids`，明确返回 copy、tuple 或只读 view。
- [ ] `is_finished`。

### Request 不变量

- [ ] request ID 不能为空。
- [ ] 首个里程碑拒绝空 prompt，并给出明确错误。
- [ ] token ID 必须为非负整数。
- [ ] `0 <= num_computed_tokens <= num_known_tokens`。
- [ ] FINISHED request 必须有 finish reason。
- [ ] 非 FINISHED request 不得有 finish reason。

### Request 状态操作

- [ ] 提供受控的 status transition 方法或集中式 transition helper。
- [ ] 提供 append sampled token 的受控操作。
- [ ] 提供增加 computed token 数的受控操作。
- [ ] 提供 finish 操作并校验合法终态。
- [ ] 非法状态迁移立即报错。

### `tests/core/test_request.py`

- [ ] 测试初始化状态和所有计算属性。
- [ ] 测试追加 output token 后 known/pending token 变化。
- [ ] 测试增加 computed token 后 pending token 变化。
- [ ] 测试 computed token 越界。
- [ ] 测试非法 token ID 和空 prompt。
- [ ] 测试合法与非法状态迁移。
- [ ] 测试 EOS、LENGTH、ABORTED finish reason。

## 7. Phase 3：Execution Plan Contract

### `src/micro_vllm/core/scheduler_output.py`

- [ ] 定义 frozen `ScheduledRequest`。
- [ ] 包含 `request_id`。
- [ ] 包含本轮输入 token IDs 的只读快照。
- [ ] 包含本轮起始 `num_computed_tokens`。
- [ ] 包含 `num_scheduled_tokens`。
- [ ] 包含是否需要采样的显式、可推导信息；避免新增持久化 phase。
- [ ] 定义 frozen `SchedulerOutput`。
- [ ] 使用不可变 scheduled request collection。
- [ ] 以 computed property 提供 `total_num_scheduled_tokens`，不存储重复计数。
- [ ] 不包含 Request 引用。
- [ ] 不包含 GPU/CPU tensor。

### Executor result contract

- [ ] 在 `src/micro_vllm/executor/interface.py` 定义最小 `ModelRunnerOutput` 或等价结果类型。
- [ ] sampled token 以 request ID 显式关联。
- [ ] 允许请求本轮没有 sampled token。
- [ ] 拒绝未知 request ID、重复 request ID 或每请求多个非预期 token。

### Executor boundary

- [ ] 定义 Engine 所需的最小 Executor protocol。
- [ ] protocol 只包含当前实际使用的 `execute(plan)`。
- [ ] 暂不加入显存探测、KV 初始化和 shutdown 等未使用方法。

### 测试

- [ ] ScheduledRequest 和 SchedulerOutput 不可修改。
- [ ] 修改原始 token list 不影响计划快照。
- [ ] total token 数计算正确。
- [ ] 非法 token 数、重复 request ID 和不一致总数被拒绝。

## 8. Phase 4：统一 Scheduler

### `src/micro_vllm/core/scheduler.py`

- [ ] Scheduler 接受 SchedulerConfig。
- [ ] waiting queue 使用确定性 FIFO 数据结构。
- [ ] running collection 保持稳定顺序。
- [ ] 提供 `add_request(request)`。
- [ ] 拒绝重复 request ID。
- [ ] 提供 `abort_request(request_id)`，集中处理 waiting/running 状态迁移。
- [ ] 提供 `get_request(request_id)` 或最小只读查询接口。
- [ ] 提供 `has_unfinished_requests()`。

### `schedule()`

- [ ] 每个 step 从完整 token budget 开始。
- [ ] 优先遍历 running requests。
- [ ] running request 最多调度其 pending token 数。
- [ ] running request 调度受剩余 token budget 限制。
- [ ] waiting request 按 FIFO admission。
- [ ] admission 受 `max_num_seqs` 限制。
- [ ] waiting request 使用剩余 token budget。
- [ ] prompt 大于剩余 budget 时只调度一个 chunk。
- [ ] budget 为零时停止继续调度。
- [ ] 没有 pending token 的非法活跃状态应被检测。
- [ ] 生成不可变 SchedulerOutput。
- [ ] `sum(num_scheduled_tokens) <= max_num_batched_tokens`。
- [ ] `len(scheduled_requests) <= max_num_seqs`。
- [ ] `schedule()` 不提前修改 `num_computed_tokens`。

### 采样边界

- [ ] 当 `start_computed + scheduled == num_known_tokens` 时，本轮请求需要采样。
- [ ] prompt 中间 chunk 不需要采样。
- [ ] 已有一个未计算 output token 的 running request 通常调度 1 token后需要采样。
- [ ] 采样判定由请求进度推导，不读取 `is_prefill`。

### `update(plan, result)`

- [ ] 只提交 plan 中实际 scheduled 的 token 数。
- [ ] 验证 result request ID 属于 plan。
- [ ] 需要采样的 request 必须得到 sampled token。
- [ ] 不需要采样的 request 不得得到 sampled token。
- [ ] 更新 `num_computed_tokens`。
- [ ] 将合法 sampled token 追加到 output token IDs。
- [ ] sampled token 等于 EOS 且未 ignore EOS 时结束请求。
- [ ] output token 数达到 max tokens 时以 LENGTH 结束。
- [ ] 未完成请求保留在 running 中。
- [ ] 完成请求从 running 中移除。
- [ ] 同一个 plan 不允许重复提交。
- [ ] update 校验失败时避免留下部分提交状态。

### `tests/core/test_scheduler.py`

- [ ] 单请求完整 prompt 在一个 step 内调度。
- [ ] 长 prompt 被切分为多个 chunk。
- [ ] 中间 prompt chunk 不要求 sampled token。
- [ ] running decode 优先于 waiting prefill。
- [ ] decode 和 prefill 进入同一个 plan。
- [ ] token budget 永不超限。
- [ ] sequence budget 永不超限。
- [ ] waiting queue 保持 FIFO。
- [ ] running 顺序保持稳定。
- [ ] duplicate request ID 被拒绝。
- [ ] 缺失、额外或重复 executor output 被拒绝。
- [ ] EOS 和 ignore EOS 行为正确。
- [ ] max tokens 行为正确。
- [ ] plan 不能重复 update。

## 9. Phase 5：Engine Control Loop

### `src/micro_vllm/engine/output.py`

- [ ] 定义最小 `RequestOutput`。
- [ ] 包含 request ID。
- [ ] 包含新增或累计 output token IDs，首版选择一种并文档化。
- [ ] 包含 finished 状态和可选 finish reason。
- [ ] 不包含 tokenizer 文本。

### `src/micro_vllm/engine/engine.py`

- [ ] Engine 构造函数显式接收 Scheduler 和 Executor。
- [ ] 提供 tokenized `add_request()`。
- [ ] Engine 或独立 ID generator 生成稳定 request ID；首版优先允许调用方传入 ID。
- [ ] 提供 `step()`。
- [ ] `step()` 严格执行 `schedule -> execute -> update`。
- [ ] 空 schedule 不调用 Executor。
- [ ] 提供 `has_unfinished_requests()`。
- [ ] 提供 `abort_request(request_id)`。
- [ ] abort waiting request。
- [ ] abort running request。
- [ ] abort 不存在或已完成 request 时行为明确。
- [ ] Engine 不直接修改 Scheduler 内部 collection。

### FakeExecutor

- [ ] 仅在 `tests/` 内实现。
- [ ] 只读取 SchedulerOutput。
- [ ] 不持有或修改 Request 引用。
- [ ] 对需要采样的 request 返回确定性 token。
- [ ] 支持按 step/request 配置返回 EOS。
- [ ] 记录接收的 plans，便于测试执行顺序。

### `tests/integration/test_engine_loop.py`

- [ ] 单请求从 WAITING 到 FINISHED。
- [ ] prompt length=10、budget=4 时调度为 4/4/2，再产生首 token。
- [ ] running decode=1、waiting prompt=10、budget=4 时计划为 1/3。
- [ ] 达到 max tokens 后停止调用 Executor。
- [ ] EOS 正确结束请求。
- [ ] ignore EOS 时继续生成。
- [ ] Engine 运行中可动态添加请求。
- [ ] waiting 和 running request 均可 abort。
- [ ] 空 schedule 不执行 FakeExecutor。
- [ ] Executor 返回非法结果时不静默污染状态。

## 10. Phase 6：公开导出与文档同步

- [ ] 从适当的 package `__init__.py` 导出稳定公共类型。
- [ ] 不导出内部 queue、transition helper 等实现细节。
- [ ] 为 Request、SchedulerOutput、Executor protocol 和 Engine 添加简洁 docstring。
- [ ] 在 V1 需求清单中只勾选本里程碑实际完成且已验证的条目。
- [ ] 不将 FakeExecutor 作为 example 或公共 API。

## 11. 推荐实现提交顺序

1. [ ] `build: add minimal Python package and pytest configuration`
2. [ ] `feat: add request and sampling domain models`
3. [ ] `feat: define scheduler and executor contracts`
4. [ ] `feat: implement token-budget scheduler`
5. [ ] `feat: add engine control loop with fake-executor tests`
6. [ ] `docs: update completed V1 control-loop requirements`

提交拆分用于控制 review 范围；如果实际实现规模较小，可以合并相邻步骤，但每一步都应保持测试可运行。

## 12. 本里程碑明确不实现

- [ ] 不实现 BlockPool、BlockTable 和 KVCacheManager。
- [ ] 不实现 prefix caching、preemption 和 resume。
- [ ] 不实现 Torch、CUDA 和 physical KV tensor。
- [ ] 不实现 Worker、ModelRunner 和 SingleProcessExecutor 的真实逻辑。
- [ ] 不实现 tokenizer 和字符串 prompt。
- [ ] 不实现 `LLM.generate()`。
- [ ] 不实现 temperature、top-k、top-p sampling。
- [ ] 不实现模型注册、权重加载和真实模型。
- [ ] 不实现 attention backend。
- [ ] 不实现 trace/metrics 系统，只允许测试检查 execution plan。
- [ ] 不实现异步 API 和 streaming text。

## 13. 最终验证

- [ ] 运行全部 CPU unit tests。
- [ ] 运行 control-loop integration tests。
- [ ] 运行格式化、lint 和 type check；若工具尚未配置，明确记录未运行项。
- [ ] 运行 `python -m compileall`。
- [ ] 检查无 Torch/CUDA 等非必要依赖。
- [ ] 检查生产源码中不存在 FakeExecutor。
- [ ] 检查 Scheduler 中不存在 prefill/decode 双路径状态机。
- [ ] 检查工作区只包含当前里程碑相关变更。
