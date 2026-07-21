# V1.4E Sampling 与 Tokenizer-backed Offline API 实现清单

## 1. 目标

在现有 tokenized Engine、ModelRunner-owned 模型/KV 生命周期和单卡执行链路之上，
完成 request-local sampling，并提供同步、批量、字符串 prompt 的 `LLM.generate()`：

```text
str prompts
    -> LLM-owned tokenizer encode
    -> tokenized Engine requests
    -> SchedulerOutput carries immutable sampling snapshots
    -> ModelRunner computes sampling-row logits
    -> Sampler applies temperature -> top-k -> top-p -> multinomial
    -> Engine commits sampled token IDs
    -> LLM-owned tokenizer decode
    -> final text/token outputs in input order
```

本阶段保留 Engine 作为 token-level control loop；`LLM` 是构造 model/tokenizer/worker/
scheduler/executor 并驱动 Engine 的高层离线 facade，不把字符串、tokenizer 或 decode
逻辑下沉到 Scheduler、Worker 或 ModelRunner。

## 2. 范围

### 本阶段实现

- [ ] greedy、temperature、top-k、top-p sampling。
- [ ] 每请求 seed 和与 batch interleaving 无关的确定性随机流。
- [ ] sampling params 的不可变执行计划快照。
- [ ] tokenizer-backed 同步批量 `LLM.generate()`。
- [ ] 文本、prompt/output token IDs、finish reason 的最终输出。
- [ ] 本地 tokenizer fixture 与固定公开 Llama/Qwen checkpoint 的离线 API 验收。

### 本阶段不实现

- beam search、best-of、`n > 1` 或 parallel sampling。
- repetition/frequency/presence penalty、min-p、typical-p 或 logits processor。
- logprobs、prompt logprobs 或 token-level score 输出。
- chat template、messages 输入或 reasoning parser；只接受原始字符串 prompt。
- streaming/async iterator；Engine 逐 step 接口继续作为后续 streaming 基础。
- detokenization 的增量状态机；本阶段只解码最终 output token IDs。
- tokenizer remote code、动态 plugin 或任意 Python tokenizer implementation。
- 修改 ModelRunner 对模型初始化、warmup/profile、KV capacity 和 physical cache 的所有权。

## 3. 核心设计决策

### 3.1 SamplingParams 与执行计划

`SamplingParams` 保持 frozen request value object，并增加：

- `temperature: float = 0.0`：零表示严格 greedy；正值启用 multinomial。
- `top_k: int | None = None`：`None` 表示不过滤。
- `top_p: float = 1.0`：范围 `(0, 1]`。
- `seed: int | None = None`：显式 seed 提供可重复 request-local stream。

过滤顺序固定为 `temperature -> top-k -> top-p -> multinomial`。greedy 模式不消费
RNG；`temperature=0` 与非默认 top-k/top-p 同时出现时拒绝，而不是静默忽略。

Scheduler 仍拥有 Request 和 lifecycle；只有当 request 在当前 step 需要 sampling 时，
`ScheduledRequest` 才携带 frozen `SamplingParams` 和 `sample_index`。ModelRunner/Sampler
不访问 Request，也不根据 batch row 猜测参数。

### 3.2 Request-local RNG 与 plan retry

随机结果必须只由 `seed + sample_index` 定义，不能依赖同 batch 中其他 request 的数量、
顺序或是否被抢占。Sampler 为每个 sampling row 创建 request-local generator；显式
`sample_index` 使 pending plan 重试产生同一 token，并避免在 ModelRunner 中维护无法随
finish/abort 清理的 request RNG 字典。

未指定 seed 时，Request admission 一次性生成并保存 effective request seed；因此不同
request 默认仍具有非确定性，但同一 pending plan retry 保持稳定。测试只对用户显式
seed 宣称跨独立运行、batching 和 preemption 可重复。

### 3.3 Tokenizer 与 LLM 所有权

- `LLM` 使用 `transformers.AutoTokenizer`，因此将 `transformers>=4.51,<5` 提升为
  runtime dependency；不自己重写 BPE/SentencePiece 或 HF tokenizer resolution。
- tokenizer 默认与 model 使用同一 path/repo/revision/cache/token/local-only 参数。
- `LLM` 负责 encode/decode、默认 EOS 注入、内部 request ID 和输入顺序恢复。
- Engine 继续只接受 token IDs 和 `SamplingParams`。
- Worker 继续只编排 ModelRunner；不持有 tokenizer、字符串或 sampling policy。
- ModelRunner 继续只构造 device input、计算 logits 并调用 Sampler。

### 3.4 同步批量 API

首版公共形态：

```python
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    revision="<commit>",
    device="cuda",
    dtype=torch.float16,
)
outputs = llm.generate(
    ["Hello", "Explain paged attention"],
    SamplingParams(max_tokens=16, temperature=0.7, top_p=0.8, seed=0),
)
llm.shutdown()
```

`generate()` 同时接受单个字符串或字符串序列；返回值统一为 tuple，并保持输入顺序。
单个 `SamplingParams` 可广播到 batch，也可传与 prompts 等长的参数序列。空 prompt、
空 batch、长度不匹配和 tokenizer 产生空 token 序列明确失败。

最终 `GenerationOutput` 至少包含：稳定 request ID、原始 prompt、prompt token IDs、
output text、output token IDs 和 `FinishReason`。它是 frozen snapshot，不暴露 Scheduler
内部 Request。

## 4. Phase 0：Sampling 参数契约

- [x] 增加 temperature、top-k、top-p 和 seed 字段及简单默认值。
- [x] 拒绝 bool、NaN/inf、负 temperature、非法 top-k/top-p/seed。
- [x] 明确 greedy 与 stochastic mode 判定。
- [x] 保持现有只传 `max_tokens` 的调用完全兼容。
- [x] 参数对象 frozen，不能在 request admission 后被外部修改。
- [x] 单元测试覆盖边界值和非法组合。

## 5. Phase 1：Sampling 快照与算法

- [x] `ScheduledRequest` 仅在 sampling step 携带 params 和 sample index。
- [x] prefix hit、chunked prefill、decode 和 preemption 后 sample index 正确。
- [x] warmup plan 不伪造用户 sampling state。
- [x] Sampler 按 row 校验 request ID、params 和 sample index 数量。
- [x] greedy 与当前 argmax 结果完全一致。
- [x] temperature scaling 与 PyTorch reference 对拍。
- [x] top-k 保留正确候选集合并处理 `k >= vocab_size`。
- [x] top-p 始终保留至少一个 token，并与排序 cumulative reference 对拍。
- [x] mixed greedy/stochastic batch 正确采样。
- [x] 相同 seed/sample index 结果一致，不同 seed 可产生不同序列。

## 6. Phase 2：Engine 随机语义

- [x] 同一 seeded request 在独立、mixed 和不同 token budget 下输出一致。
- [x] seeded request 不受同 batch 其他 request 加入/结束影响。
- [x] chunked prefill 不提前消费 RNG。
- [x] preemption/recompute 不重复或跳过 sample index。
- [x] pending plan retry 返回同一 sampled token。
- [x] EOS、ignore EOS、max tokens 和 model length 对 stochastic sampling 仍正确。
- [x] finished/aborted request 无 sampler state 泄漏。

## 7. Phase 3：Tokenizer Adapter

- [x] Transformers 成为 tokenizer-backed API 的 runtime dependency。
- [x] tokenizer source 接受与 model source 相同的 path/repo/revision/cache/offline 参数。
- [x] 定义 LLM 实际消费的窄 tokenizer protocol，测试可注入 deterministic fake。
- [x] encode 返回自有 token ID tuple，decode 只处理 output IDs。
- [x] tokenizer EOS 通过高层 helper 注入，Engine 的 `None` 语义不改变。
- [x] public checkpoint 支持 `local_files_only=True` 完全离线 tokenizer reload。
- [x] 不使用 `trust_remote_code`。

## 8. Phase 4：LLM.generate Offline API

- [ ] `LLM` 构造并拥有 tokenizer、Worker、Scheduler、Executor 和 Engine。
- [ ] 模型仍通过延迟 initializer 只由 ModelRunner 构造与持有。
- [ ] 单 prompt 和 batch prompt 保持输入顺序。
- [ ] 单个 sampling params 广播和 per-prompt params 均受支持。
- [ ] 同一 LLM 实例支持多次 sequential `generate()`。
- [ ] 每次 batch request ID 唯一且不与历史 Scheduler request 冲突。
- [ ] `GenerationOutput` 文本/token/final reason 完整且不可变。
- [ ] `shutdown()` 幂等，并释放 Worker/ModelRunner 引用。
- [ ] context-length admission、空输入和 tokenizer 错误具有清晰上下文。

## 9. Phase 5：真实模型验收与文档

- [ ] tiny local tokenizer + Llama/Qwen checkpoint 完成字符串到文本闭环。
- [ ] 固定 TinyLlama commit 的 greedy 文本/token IDs 与 Transformers 一致。
- [ ] 固定 Qwen3 commit 的 seeded sampling 在 repeated/mixed batch 中可重复。
- [ ] 在线 tokenizer 首次下载后可用 `HF_HUB_OFFLINE=1` 完全离线生成。
- [ ] 记录公开模型、revision、prompt、sampling params、tokens 和输出文本。
- [ ] sampling/LLM 定向、非 model 和完整默认回归通过。
- [ ] 更新 public exports、README/requirements 和最小使用示例。

## 10. 验证命令

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/sampling tests/llm'

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m "not model"'

ssh aliyun-ecs \
  'cd /root/micro-vllm && MICRO_VLLM_TEST_MODEL=1 HF_HUB_OFFLINE=1 \
   .venv/bin/python -m pytest -q -m model tests/llm'

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q'
```

## 11. 完成标准

- [ ] temperature/top-k/top-p 具有独立 reference correctness tests。
- [ ] seeded request 在 batching、chunking 和 preemption 变化下保持确定性。
- [ ] Scheduler/Engine 不执行 logits filtering，Sampler 不访问 Request。
- [ ] tokenizer/string 逻辑只存在于 LLM 高层。
- [ ] `LLM.generate()` 可在单 GPU 上加载两个真实模型族并返回最终文本。
- [ ] ModelRunner 继续独占模型初始化、warmup/profile 和 KV cache 生命周期。
- [ ] 普通回归无网络，公开 tokenizer/model 验收显式启用并可完全离线。
- [ ] `docs/requirements/v1-core-engine.md` 按实际证据更新。

## 12. 实现状态

2026-07-21 已完成 Phase 0–3。`SamplingParams` 新增 temperature、top-k、top-p、
seed 和明确的 `is_greedy` 判定，同时保持现有 `max_tokens`/EOS 调用兼容。
`ScheduledRequest` 在 sampling boundary 携带 frozen params/sample index；ModelRunner
使用通用 Sampler 完成 greedy、temperature、top-k 和 top-p，warmup 不伪造用户参数。
Phase 2 进一步验证 seeded request 在 independent/mixed、不同 token budget、chunked
prefill 和 preemption 下序列一致；unseeded request 在 admission 时固化 effective seed，
pending retry 不换 token，Sampler 不保存 request state。定向回归为 `95 passed`，完整
默认回归为 `522 passed, 4 skipped`。

Phase 3 将 Transformers 4.51+ 提升为 runtime dependency，增加窄 `Tokenizer` protocol、
tuple-owning adapter、HF source 参数转发和高层 EOS helper。固定 Qwen3 revision 首次
tokenizer 加载为 10.463s，cache 重载约 0.5–0.7s，`HF_HUB_OFFLINE=1` 验收通过；
固定 prompt tokens 为 `[9707, 11, 281, 3279, 6529, 0]`。tokenizer 定向回归为
`4 passed, 1 skipped`，完整默认回归为 `526 passed, 5 skipped`。Phase 4
`LLM.generate()` 尚未实现。
