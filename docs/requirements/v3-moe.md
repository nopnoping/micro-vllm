# V3：MoE 需求清单

## 1. 版本目标

在 V2 分布式能力上实现推理态 Mixture-of-Experts，保留动态 token routing、expert 执行、token permutation 和 Expert Parallel 的核心概念，不引入训练态负载均衡或生产级动态专家管理。

推荐实现顺序：

1. 单 GPU reference MoE；
2. grouped/fused expert execution；
3. Expert Parallel；
4. TP、PP、EP 组合。

## 2. 公共抽象

### P0

- [ ] 定义 `Router`，只负责从 hidden states 生成 routing decision。
- [ ] 定义 `RoutingPlan`，包含 token index、expert ID、routing weight 和 top-k 信息。
- [ ] 定义 `ExpertExecutor`，消费 permuted tokens 和 RoutingPlan。
- [ ] 定义 `ExpertPlacement`，映射 global expert ID 到 EP rank/local expert ID。
- [ ] routing、dispatch、expert compute、combine 四阶段边界清晰。
- [ ] 模型类只组装 MoE layer，不包含分布式 dispatch 细节。

建议数据结构：

```python
@dataclass
class RoutingPlan:
    expert_ids: Tensor
    expert_weights: Tensor
    token_indices: Tensor
    tokens_per_expert: Tensor
```

## 3. 单 GPU Reference MoE

### P0

- [ ] 实现 router logits 计算。
- [ ] 实现 top-k expert 选择。
- [ ] 实现 routing weight normalization。
- [ ] 实现 token 按 expert permutation。
- [ ] 实现每个 expert 的 gate/up/down MLP。
- [ ] 实现 expert output 按原 token 顺序 combine。
- [ ] 支持 shared expert。
- [ ] 支持某个 expert 本轮没有 token。
- [ ] inference 不丢 token，不使用 capacity factor。
- [ ] 提供逐 expert Python/reference 实现用于对拍。

### P1

- [ ] 支持 grouped GEMM 或等价 optimized expert backend。
- [ ] 支持 fused permutation/unpermutation。

## 4. 模型与权重加载

- [ ] Model Registry 支持至少一个代表性 MoE 模型族。
- [ ] 推荐首个模型选择 Mixtral 或结构清晰的 Qwen MoE。
- [ ] 支持 routed expert 和 shared expert 权重命名映射。
- [ ] 支持 packed expert 参数加载。
- [ ] WeightLoader 根据 ExpertPlacement 只加载本 rank 的 expert。
- [ ] PP partition 后只有包含该 MoE layer 的 stage 加载对应 expert。
- [ ] checkpoint expert 数与 config 不一致时明确报错。

## 5. Expert Parallel

### Topology

- [ ] `ParallelContext` 增加 EP group 表达能力。
- [ ] V3 首版使用显式 EP group，不隐式复用 DP 或 TP rank。
- [ ] dense layer 在 EP group 上采用复制或明确的 TP 策略。
- [ ] expert 在 EP group 内按 ExpertPlacement 分片。
- [ ] 校验 expert 数、EP size 和 placement 的合法性。

### Dispatch 与 Combine

- [ ] 根据目标 expert rank 统计 send counts。
- [ ] 实现 token hidden states 的 all-to-all dispatch。
- [ ] 同步传输恢复顺序所需的 token/routing metadata。
- [ ] local expert 执行完成后 all-to-all 返回结果。
- [ ] 根据原 token index 和 top-k slot 合并输出。
- [ ] 支持不同 rank 本轮 token 数不均衡。
- [ ] 支持某 rank 本轮接收零 token。
- [ ] 通信 tensor 的 dtype/device 保持一致。

### 组合并行

- [ ] EP 与 PP 可组合，EP group 不跨错误的 PP stage。
- [ ] EP 与 TP 的权重分片规则有明确限制或实现。
- [ ] DP replica 默认相互独立，不共享正在执行的 request。
- [ ] Scheduler 仍然按 token 数调度，不感知 token 被路由到哪个 expert。

## 6. 可观察性

- [ ] 每层记录 tokens per expert。
- [ ] 记录 expert load imbalance 指标。
- [ ] 记录每个 EP rank 的 send/recv token 数。
- [ ] 记录 dispatch、expert compute 和 combine 耗时。
- [ ] trace 可以将 token 映射到 top-k experts。
- [ ] 指标默认只保存统计量，不持久化完整 hidden states。

## 7. 测试

- [ ] top-1 和 top-2 routing 分别有 reference test。
- [ ] permutation/unpermutation 能恢复原 token 顺序。
- [ ] routing weight combine 与朴素实现对拍。
- [ ] 零 token expert/rank 不死锁。
- [ ] 单 GPU optimized expert 与 reference MoE 数值对拍。
- [ ] EP=1 与 EP>1 输出对拍。
- [ ] MoE + TP、MoE + PP 至少各有一个端到端测试。
- [ ] chunked prefill 和 decode mixed batch 可通过 MoE layer。
- [ ] 不同 batch token 分布不会导致 shape 或 collective 错误。
- [ ] 权重加载只读取本 rank 所需 expert。

## 8. 明确不包含

- 训练态 auxiliary load-balancing loss；
- capacity factor 和 token drop；
- dynamic expert placement；
- expert load balancing 在线迁移；
- redundant expert/hot expert replication；
- 通信计算深度 overlap；
- DeepSeek 等所有 MoE 变体兼容；
- 专有 fused kernel 的完整集合。

## 9. V3 发布验收

- [ ] 单 GPU MoE 与 dense/reference 计算路径完成数值验证。
- [ ] 至少一个真实 MoE checkpoint 可以加载并生成文本。
- [ ] EP all-to-all 在 token 分布极不均衡和零 token 情况下正确结束。
- [ ] EP=1 与 EP>1 greedy 输出在容许误差内一致。
- [ ] 提供 expert load trace 和通信/计算 breakdown。
- [ ] V1 Scheduler 不因 MoE 引入模型特定分支。
