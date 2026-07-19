# V5：投机解码需求清单

## 1. 版本目标

在 V1 的统一 token accounting 上实现 speculative decoding：由 drafter 提议多个 token，target model 一次验证，并在接受或拒绝后正确更新请求进度与 KV cache。

推荐实现顺序：

1. n-gram drafter；
2. greedy acceptance；
3. draft model；
4. stochastic rejection sampling；
5. variable proposal length batching。

## 2. 核心原则

- draft token 是“已知但尚未由 target model 计算”的 token。
- Scheduler 继续使用 `num_known_tokens - num_computed_tokens` 统一计算 pending token。
- 不增加独立 speculative queue 或 speculative phase 状态机。
- target verification 结果通过 `ModelRunnerOutput` 返回，Engine 统一提交或回滚。
- KV rollback 是投机解码正确性的核心组成部分，而不是事后修补。

## 3. 公共抽象

### P0

- [ ] 定义 `Drafter` protocol。
- [ ] 定义 `DraftOutput`，包含每请求 proposal token IDs 和可选 proposal probabilities。
- [ ] 定义 `Verifier` 或 verification result 数据结构。
- [ ] 定义 `AcceptanceSampler`，区分 greedy 和 stochastic acceptance。
- [ ] `SpeculativeConfig` 能配置 drafter、最大 draft token 数和 acceptance 模式。
- [ ] speculative 功能关闭时完全回到 V1 正常 decode 路径。

建议接口：

```python
class Drafter(Protocol):
    def propose(self, requests, max_tokens: int) -> DraftOutput: ...

@dataclass
class VerificationResult:
    accepted_token_ids: dict[str, list[int]]
    replacement_token_ids: dict[str, list[int]]
    num_accepted: dict[str, int]
```

## 4. N-gram Drafter

### P0

- [ ] 从请求已有 token 序列中查找 n-gram match。
- [ ] 最多返回配置数量的 proposal tokens。
- [ ] 无匹配时返回空 proposal，不影响正常 decode。
- [ ] drafter 不访问 GPU model weights。
- [ ] proposal 生成过程可通过 trace 解释匹配位置和 token 数。

## 5. Target Verification

### P0

- [ ] Scheduler 可一次为请求计划多个 draft token。
- [ ] ModelRunner 能为连续多个 verification position 计算 target logits。
- [ ] 支持同一 batch 中不同请求 proposal 长度不同。
- [ ] attention metadata 正确包含 draft token 的连续 positions。
- [ ] target KV 按计划写入，但在结果提交前可安全回滚。
- [ ] 末尾额外 target token 的生成语义有明确实现和测试。
- [ ] verification 不破坏同时存在的普通 decode/chunked prefill 请求。

## 6. Greedy Acceptance

### P0

- [ ] 从第一个 proposal token 开始逐个比较 target argmax。
- [ ] 连续匹配的 token 被接受。
- [ ] 第一个不匹配位置使用 target token 替换并停止接受后续 proposal。
- [ ] 全部 proposal 接受时追加 target 预测的额外 token。
- [ ] EOS 出现在 accepted/replacement token 中时正确终止请求。
- [ ] max tokens/max model length 可截断 accepted sequence。

## 7. Stochastic Rejection Sampling

### P1

- [ ] DraftOutput 提供 proposal probability。
- [ ] Target verification 保留对应 token distribution 所需信息。
- [ ] 按正确 acceptance probability 接受 proposal。
- [ ] 拒绝时从修正分布采样 replacement token。
- [ ] per-request RNG 可复现。
- [ ] temperature、top-k、top-p 与 speculative sampling 的作用顺序有明确规定。
- [ ] 小词表模型上通过分布统计测试，而不只验证单次 token。

## 8. Draft Model

### P0/P1

- [ ] 实现 `DraftModelDrafter`。
- [ ] draft model 拥有独立 model runner 和 KV cache。
- [ ] draft model tokenizer/vocab 必须与 target 兼容，或启动时拒绝。
- [ ] draft model 的 request 生命周期与 target request 对齐。
- [ ] target 接受/拒绝后同步提交或回滚 draft KV 状态。
- [ ] draft model 可以使用独立 token budget 或明确的串行策略。
- [ ] 分布式 target 与单卡 draft 的拓扑限制被校验。

## 9. KV Cache 提交与回滚

### P0

- [ ] KVCacheManager 支持按 token position 截断 request block table。
- [ ] 被拒绝 token 所属的完整尾部 block 可释放。
- [ ] partial block 中无效 slot 不被后续 prefix cache 视为有效内容。
- [ ] rollback 后 `num_computed_tokens` 与已接受 token 数严格一致。
- [ ] rollback 不修改仍被其他 request 引用的共享 prefix block。
- [ ] 只有完成并提交的完整 block 可进入 PrefixCache。
- [ ] rollback、preemption 和 request finish 的组合不会 double free。
- [ ] local/sliding cache group 使用各自 retention policy 正确回滚。

## 10. Scheduler 集成

- [ ] draft token 与普通 pending token 使用同一个 token budget。
- [ ] 单请求 verification token 数不得独占整个 batch 而造成无限饥饿。
- [ ] mixed batch 支持普通 decode、spec verification 和 chunked prefill。
- [ ] SchedulerOutput 标明每请求哪些 scheduled tokens 属于 proposal。
- [ ] Engine update 根据 acceptance 数修正 `num_computed_tokens`。
- [ ] 请求被抢占时清理尚未提交的 proposal state。
- [ ] 请求取消时同时释放 target 和 draft 资源。

## 11. 可观察性与指标

- [ ] 记录 proposed、accepted、rejected token 数。
- [ ] 记录 acceptance length histogram 和 acceptance rate。
- [ ] 记录 target forward 次数和每次验证 token 数。
- [ ] 记录 draft latency、verification latency 和 rollback 次数。
- [ ] benchmark 比较关闭/开启 speculative 的吞吐、ITL 和显存。
- [ ] trace 可展示一次 proposal -> verify -> commit/rollback 全过程。

## 12. 测试

- [ ] 全接受、首 token 拒绝、中间拒绝和空 proposal 分别测试。
- [ ] proposal 中出现 EOS、超出 max tokens、跨 block boundary 分别测试。
- [ ] greedy speculative 输出与普通 greedy decode 完全一致。
- [ ] rollback 后继续生成与从正确 prefix 重跑结果一致。
- [ ] prefix cache + speculative 不缓存未提交 token。
- [ ] preemption + speculative 无 block 泄漏。
- [ ] mixed batch 中请求互不污染。
- [ ] n-gram 和 draft model drafter 均有端到端测试。
- [ ] stochastic 模式通过小模型分布正确性检验。
- [ ] TP/PP 下至少验证 greedy speculative 路径。

## 13. 明确不包含

- EAGLE；
- MTP；
- Medusa；
- tree-based speculative decoding；
- 多级 drafter cascade；
- 面向每个模型的动态 draft length 学习器；
- 生产级异步 draft/target overlap。

## 14. V5 发布验收

- [ ] n-gram speculative greedy 路径可运行且输出与普通 greedy 完全一致。
- [ ] draft model 路径能够完成 proposal、verification 和 rollback。
- [ ] variable proposal length mixed batch 正确执行。
- [ ] 跨 partial/full block 的拒绝场景无无效 prefix cache 或 block 泄漏。
- [ ] 至少一种配置在代表性 workload 上减少 target forward 次数或改善吞吐。
- [ ] Scheduler 未引入独立 prefill/decode/speculative 三套核心路径。
