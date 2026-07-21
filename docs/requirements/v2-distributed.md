# V2：分布式推理需求清单

## 1. 版本目标

在不改变 V1 Engine/Scheduler 核心语义的前提下，引入 Tensor Parallel、Data Parallel 和 Pipeline Parallel，验证控制面计划如何被多个设备进程一致执行。

推荐实现顺序：

1. V2.1 Tensor Parallel；
2. V2.2 Data Parallel；
3. V2.3 Pipeline Parallel；
4. V2.4 Pipeline batch overlap。

## 2. 并行拓扑与公共抽象

### P0

- [ ] 定义 `ParallelConfig(dp_size, pp_size, tp_size)`。
- [ ] 定义 rank 到 `(dp_rank, pp_rank, tp_rank)` 的稳定映射。
- [ ] 定义 `ParallelContext`，集中管理 rank、world size、device 和 process groups。
- [ ] 建立 TP、PP、DP 各自独立的 process group。
- [ ] 明确拓扑为 `DP replica -> PP stage -> TP ranks`。
- [ ] 配置验证 world size 与设备数一致。
- [ ] 所有 collective 封装在 `distributed/collectives.py` 或等价模块中。
- [ ] 模型 layer 不直接创建 process group。
- [ ] 单卡配置继续使用同一套接口，且不初始化不必要的 collective。
- [ ] 分布式初始化失败时所有子进程能够退出，不遗留挂起进程。

### P1

- [ ] trace 中记录每个 collective 的 group、tensor shape 和耗时。
- [ ] 支持使用 `torchrun` 外部启动，内部语义与 multiprocessing 启动一致。

## 3. Multiprocess Executor

### P0

- [ ] 在 V1 `Executor` protocol 下实现 `MultiprocessExecutor`。
- [ ] 每个 GPU 对应独立 Worker 进程。
- [ ] Engine 进程不持有模型权重和 CUDA context。
- [ ] Executor 负责 Worker 生命周期、命令广播、结果收集和 shutdown。
- [ ] 定义最小 Worker command 集：初始化、加载模型、探测内存、初始化 KV、执行、停止。
- [ ] 传输 SchedulerOutput 时避免复制无关 Request Python 对象。
- [ ] Worker 异常能够被 Executor 检测并报告原始 traceback。
- [ ] 所有 Worker 对 execution plan 的 request 顺序和 block ID 达成一致。
- [ ] 初始化 KV cache 时使用所有相关 rank 可用容量的安全下界。
- [ ] shutdown 具备幂等性。

### 非目标

- [ ] 不实现 Ray backend。
- [ ] 不实现生产级 fault tolerance 和 worker restart。
- [ ] 不实现跨节点 elastic scaling。

## 4. Tensor Parallel

### Layer

- [ ] 实现 Column Parallel Linear。
- [ ] 实现 Row Parallel Linear。
- [ ] 实现 vocab-parallel embedding。
- [ ] 实现 vocab-parallel LM head。
- [ ] 支持 attention head 和 KV head 的 TP 分片。
- [ ] 明确处理 GQA/MQA 中 KV head 数小于 TP size 的限制或复制策略。
- [ ] 支持 packed QKV 在 TP rank 上正确切分。
- [ ] 支持 packed gate/up 在 TP rank 上正确切分。
- [ ] collective 调用由 layer abstraction 触发，而非模型类散落调用。

### Weight Loading

- [ ] Model Loader 的 checkpoint 边界继续提供 tensor stream，不解释 TP parameter layout。
- [ ] 模型或并行 layer 的 `load_weights()` 根据 TP rank 只复制/切分本 rank 所需 tensor。
- [ ] row/column/vocab shard 规则由拥有 parameter layout 的模型/layer 声明。
- [ ] shape 校验使用 global shape 和 local shape 两套信息。
- [ ] tied embedding 在 vocab parallel 下保持一致。

### Execution

- [ ] TP ranks 接收完全相同的 SchedulerOutput。
- [ ] TP ranks 使用一致的 token positions、slot mapping 和 block IDs。
- [ ] sampling 只在指定 rank 执行，结果广播给其他必要 rank。
- [ ] rank 间随机数状态不会导致 token 分叉。

### 测试

- [ ] TP=1 与 TP=2/4 的 greedy token 输出一致。
- [ ] 单层 Column/Row Parallel Linear 与 dense linear 对拍。
- [ ] GQA、packed QKV 和 vocab parallel 分别具有单元测试。
- [ ] TP 下 prefix cache、preemption、chunked prefill 继续通过 V1 测试。

## 5. Data Parallel

V2 中 dense model 的 DP 定义为 serving replica parallelism：每个 replica 拥有独立 Engine、Scheduler、模型副本和 KV cache。

### DP Coordinator / Router

- [ ] 定义 DP replica 的注册和健康状态。
- [ ] 请求只能属于一个 DP replica，生命周期中默认不迁移。
- [ ] 每个 replica 独立维护 Scheduler 和 KV cache。
- [ ] 提供 round-robin 路由策略。
- [ ] 提供基于 active requests/token backlog 的负载路由策略。
- [ ] 路由指标获取不得阻塞每次模型执行。
- [ ] request ID 在全局范围唯一，输出可正确汇聚到调用方。
- [ ] 一个 replica 满载时新请求可路由到其他 replica。
- [ ] DP 不要求 token step 级同步。

### P1

- [ ] 路由策略可考虑 KV block 使用率和预计 prompt token 数。
- [ ] 支持为同一会话提供 replica affinity，提高 prefix cache 命中概率。

### 测试

- [ ] DP=1 与 DP>1 单请求输出一致。
- [ ] 多 replica 请求分布符合配置的路由策略。
- [ ] 每个 replica 的 block pool 完全隔离。
- [ ] 单个 replica 忙时其他 replica 仍可接收请求。

## 6. Pipeline Parallel

### Model Partition

- [ ] 根据 PP rank 将 Transformer layers 连续切分到不同 stage。
- [ ] 首 stage 持有 input embedding。
- [ ] 末 stage 持有 final norm、LM head 和 sampler。
- [ ] tied weights 跨首末 stage 时有明确策略：复制或通信，V2 推荐复制。
- [ ] 每个 stage 只加载自身 layer 权重。
- [ ] 每个 stage 只分配自身 attention layer 所需 KV tensor。

### Runtime

- [ ] 定义 stage 间中间 tensor 的数据结构和 shape metadata。
- [ ] 实现相邻 PP stage 的 send/recv。
- [ ] 所有 stage 对 request 顺序、scheduled token 数和 positions 保持一致。
- [ ] 末 stage 将 sampled token 返回 Engine。
- [ ] 首版允许每次只存在一个 in-flight batch。
- [ ] PP stage 可包含 TP group，通信边界和 rank 角色明确。
- [ ] Engine 初始化时汇总各 stage KV cache spec 和可用容量。
- [ ] Scheduler 使用所有 stage 可满足的公共 KV 容量下界。

### Batch Overlap（P1）

- [ ] 引入有界 in-flight batch queue。
- [ ] 每个 SchedulerOutput 具有单调递增 batch/step ID。
- [ ] 不同 step 的返回结果能够按 ID 与原计划匹配。
- [ ] queue 满时 Engine 停止继续调度，形成清晰 backpressure。
- [ ] overlap 开关关闭后回退为同步 PP 路径。

### 测试

- [ ] PP=1 与 PP=2/4 输出一致。
- [ ] layer partition 不重不漏。
- [ ] 每个 stage 只加载预期参数。
- [ ] PP+TP 组合与单卡 reference 对拍。
- [ ] chunked prefill、prefix cache、preemption 在 PP 下语义不变。
- [ ] 多 in-flight batch 不会交换 request output 或覆盖 block table。

## 7. 配置与可观察性

- [ ] 启动日志打印全局 rank 与 dp/pp/tp rank。
- [ ] 打印每个 process group 成员。
- [ ] 打印每个 PP stage 的 layer 范围和参数量。
- [ ] 打印每 rank 权重显存和 KV cache 容量。
- [ ] benchmark 分别记录 TP collective、PP send/recv 和模型计算时间。
- [ ] DP router 记录每个 replica 的请求数、token backlog 和路由决策。

## 8. 明确不包含

- Expert Parallel；
- context parallel；
- disaggregated prefill/decode；
- pipeline interleaving 的复杂策略；
- 跨节点容错和弹性扩缩；
- Ray；
- 动态迁移正在运行的 DP request；
- 通信/计算的深度定制融合 kernel。

## 9. V2 发布验收

- [ ] TP、DP、PP 均可单独启用。
- [ ] 至少一种 `TP > 1 && PP > 1` 组合通过端到端测试。
- [ ] DP 能将独立请求路由到多个 replica。
- [ ] 分布式输出与单卡 greedy baseline 一致或有明确浮点误差说明。
- [ ] V1 的调度和 KV cache 核心测试在分布式配置下继续通过。
- [ ] 无 Worker 静默挂起，异常可定位到具体 global/dp/pp/tp rank。
- [ ] 提供 TP、PP、DP 各一个最小运行示例和 benchmark。
