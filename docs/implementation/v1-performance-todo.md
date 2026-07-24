# V1.6 Optimized Attention 与性能收尾实现清单

## 1. 目标

在不改变 V1 已冻结的 Request、SchedulerOutput、KV cache group、ModelRunner 和
model forward contract 的前提下，为单卡执行增加一个可显式选择的 GPU optimized
attention path，建立可复现的 reference baseline、正确性对拍和端到端 benchmark，
并据实决定 `torch.compile`、streaming 和 priority 是否进入 V1。

本清单对应上一轮建议中的第 2–6 项：

```text
2. 冻结 benchmark/最小测量格式并测 reference baseline
3. 实现一个 optimized GPU attention backend
4. 完成 prefill/decode/chunked parity 和显式 fallback
5. 完成 benchmark，并评估 torch.compile
6. 决定 streaming/priority 是否进入 V1
```

最终数据流保持为：

```text
ModelConfig.attention_operator
    -> Attention creates one statically registered operator
    -> ModelRunner builds the existing per-group AttentionMetadata
    -> operator writes current K/V and reads paged KV through block_table
    -> reference or optimized tensor implementation
    -> unchanged model hidden/logits/sampling/control loop
```

## 2. 范围和非目标

### 本阶段实现

- 固定公开 Qwen3 checkpoint 的 T4 FP16 reference baseline。
- tokenized Engine benchmark，不依赖 tokenizer、网络或随机 prompt 文本。
- 一个 optimized attention operator；目标为批量执行的 Triton paged attention
  路径，PyTorch SDPA 只保留为 feasibility 对照。
- Full、SlidingWindow、ChunkedLocal 和 hybrid 的数值/生命周期回归。
- full prefill、chunked prefill、decode 和 mixed batch 的对拍。
- reference/optimized 显式切换和 benchmark 对比。
- benchmark 对比后按实际诊断需要补齐可关闭的 engine/scheduler performance trace。
- 只选择一种 graph/compile 实验；V1.6 优先评估 `torch.compile`。
- 对 streaming、priority、prefill chunk policy 做明确的 V1 范围决策。

### 本阶段不实现

- 多套 optimized backend 或自动 backend selector。
- 在 Scheduler 中增加 Full/SW/Chunked 或 operator 分支。
- 改变 physical KV cache layout、block table 或 cache group contract。
- 为 optimized operator 建立独立 ModelInput 或 prefill/decode execution plan。
- silent fallback；调用方通过 `attention_operator="reference"` 显式回退。
- 为追求 kernel 覆盖引入生产级 backend framework。
- 同时实现 CUDA Graph 和 `torch.compile`。
- 多 GPU、量化、speculative decoding 或新模型架构。
- `tokenizer_revision`。
- BlockPool 的防御性 runtime invariant validation；继续由封装和生命周期测试保证。

## 3. 冻结的设计边界

### 3.1 Operator contract 不变

optimized operator 继续实现现有接口：

```python
operator(
    query,
    key,
    value,
    layer_kv_cache,
    attention_metadata,
) -> output
```

- `AttentionMetadata` 仍包含 positions、context starts、request boundaries、
  block table、slot mapping 和 block size。
- operator 内部可以根据 query length 选择适合的 tensor 路径，但不把该选择写入
  SchedulerOutput。
- current K/V 必须先写入 metadata 指定的 physical slots；reference 和 optimized
  共享相同的 cache side effect。
- Triton operator 直接通过 block table 读取 paged history；独立 gather 只允许作为
  地址映射学习/对拍步骤，不进入最终 hot path。

### 3.2 只增加一个静态 operator

- registry 增加一个明确名称，初版建议为 `"triton_paged"`。
- `"reference"` 始终保留为正确性基线和显式 fallback。
- 不增加动态 register、entry point、capability graph 或自动硬件探测 registry。
- 不在运行中捕获 optimized 异常后静默切换 reference；不支持的外部配置在初始化/
  首次执行边界给出直接错误。

### 3.3 Benchmark 不进入 runtime

- benchmark harness 放在 `benchmarks/`，可以使用 internal Engine/Worker 组装接口。
- runtime 不为 benchmark 暴露可修改内部对象，也不增加只被脚本消费的 production API。
- latency 使用 wall-clock monotonic timer；GPU 区间在测量边界显式 synchronize。
- benchmark 结果不是单元测试，不设置容易受噪声影响的 CI 性能阈值。

## 4. Phase 0：最小 benchmark 与 reference baseline（对应第 2 项）

### 4.1 固定 workload

- [x] 使用 `Qwen/Qwen3-0.6B` revision
      `c1899de289a04d12100db370d81485cdf75e47ca`。
- [x] 使用标准 Hub cache 和 `local_files_only=True`，测量时不访问网络。
- [x] 固定 Tesla T4、FP16、reference eager、greedy sampling 和随机 seed。
- [x] 记录 PyTorch、CUDA、GPU 名称/显存、model revision、dtype、operator、block size、
      token/sequence budget 和 KV page capacity。
- [x] 至少定义以下 tokenized workloads：
  - prefill：单 request，多个固定 prompt length，输出一个 token；
  - decode：短 prompt，固定长输出；
  - mixed：running decode 与新 prefill 共处一个 batch；
  - cache pressure：小容量下触发 prefix/reuse 或 preemption。
- [x] warmup 次数、测量次数和输入 token IDs 固定并写入结果。

### 4.2 结果 schema

- [x] benchmark 输出单个 JSON document，不解析人类日志。
- [x] 配置字段包含 model/hardware/runtime/workload 的完整可复现参数。
- [x] 结果至少包含：
  - elapsed seconds；
  - request throughput；
  - prompt/output/total token throughput；
  - TTFT；
  - ITL 的 count、median 和 p90；
  - CUDA peak allocated memory；
  - physical KV bytes、page count 和 peak utilization。
- [x] 单位固定为 seconds、tokens、bytes，不在 schema 中混用格式化字符串。
- [x] stdout 可打印简短摘要；机器可读 JSON 写入调用方显式指定的路径。

### 4.3 最小测量边界

- [x] benchmark harness 直接计时 `Engine.step()`，从 RequestOutput 计算 TTFT/ITL 和
      output token throughput。
- [x] 在 benchmark 外层使用 CUDA synchronize 和 peak-memory API，不把计时器放进
      model/attention hot path。
- [x] 复用现有 `KVCacheLayoutTrace` 或初始化后的只读 layout 获取 physical KV bytes 和
      page count。
- [x] peak utilization 由 benchmark harness 对已知 workload/cache capacity 计算；若
      当前信息不足，再增加一个最小只读计数，不先建设完整 lifecycle trace。
- [x] Phase 0 不新增 engine-step、request-lifecycle 或逐 block trace event。
- [x] 某项指标无法从 harness 获得时，先记录为空并说明原因，不因此阻塞 optimized
      feasibility spike。

### 4.4 Reference baseline

- [x] 先只运行 `attention_operator="reference"`。
- [x] 保存 prefill、decode、mixed 和 cache-pressure 的原始 JSON 结果。
- [x] 重复运行确认指标没有数量级抖动；不为正常噪声设置硬编码阈值。
- [x] reference baseline 未记录前，不开始永久性的 optimized code 改造。

## 5. Phase 1：单一 optimized GPU operator（对应第 3 项）

### 5.1 小型 feasibility spike

- [x] 用当前 paged gather 输出验证 `torch.nn.functional.scaled_dot_product_attention`
      在 T4/FP16 上的可用性。
- [x] 分别测量 prefill 和 decode；记录 PyTorch 实际可用 kernel 和 graph break。
- [x] spike 不进入 registry，不保留未证明收益的通用抽象。
- [x] 若 SDPA 在目标 workload 没有可测收益，记录结果并在继续前重新 review direct
      paged kernel/依赖选择，不把无收益实现命名为 optimized backend。

  本次 spike 的两轮结果分别为 prefill `2.292x/2.292x`、decode
  `1.259x/1.257x`，因此不触发无收益 review gate，可以进入正式 operator 实现。

#### 5.1.1 RTX 5090 分阶段测量

- [x] 将 GPU 名称从 spike 实现中解耦；可用 `--expected-gpu-name` 为单次实验显式
      锁定硬件，输出仍记录实际 GPU 和 compute capability。
- [x] 在 RTX 5090 上用相同 shape、FP16、seed 和测量次数重复两轮 SDPA spike。
- [x] 隔离测量 KV write、paged gather、request scalar/view extraction、dense SDPA
      preparation、SDPA core 和 output copy。
- [x] 原始结果按 GPU 和 run 分文件保存，不覆盖或混入 T4 baseline。
- [x] isolated component 每项单独 synchronize，只用于排序优化目标，不把 medians
      相加当作完整 operator latency。

两轮最大的 isolated stage 都是 paged gather（约 `0.082 ms`），其次是 SDPA core
（prefill 约 `0.064 ms`、decode 约 `0.058 ms`）和 dense input preparation
（约 `0.051 ms`）；request scalar/view extraction 约 `0.030 ms`。因此不实现
dense-only Triton core；学习步骤从 batched page write 和 page-address mapping
开始，随后直接进入 fused ragged paged attention。

#### 5.1.2 教学型 Triton paged 路径

- [x] 第一步实现 batched Triton page-write kernel：grid 覆盖全部 flattened token、
      KV head 和 head-dim block，通过 `slot_mapping` 将 K/V 写入 physical cache；
      不按 request 启动 kernel。
- [x] 第二步实现独立 Triton page-address/gather spike，用多 request、乱序 physical
      block table 和跨 page context 对拍 reference；该 tensor materialization 只用于
      学习地址映射，不进入最终性能路径。
- [ ] 第三步实现 correctness-first 的 ragged paged-attention kernel：grid 同时覆盖
      flattened query token 和 query head，kernel 内根据 `query_start_locations`
      找到 request，直接通过该 request 的 `block_table` 加载 K/V。
- [ ] 在同一个 kernel 中完成 GQA head mapping、causal/context-start mask 和 FP32
      online softmax；不生成 gathered K/V、`repeat_interleave` 结果或 dense mask。
- [ ] mixed batch 的所有 request 在一个 kernel launch 中执行；Python/C++ 只做一次
      launch，不允许逐 request 调 SDPA 或 Triton。
- [ ] correctness-first 版本允许在 GPU 内扫描较小的
      `query_start_locations` 来完成 flattened-query 到 request 的映射；测量后再比较
      GPU request-index tensor、二维 padded grid 或 work queue，不能预先增加复杂调度。
- [ ] page write、独立 gather spike、fused paged attention 分别与 reference 对拍，
      覆盖 decode、prefill、mixed、跨 page、GQA/MQA 和非连续 physical blocks。
- [ ] 在同一 warmup/measurement loop 中交错测量完整 reference、SDPA spike 和
      Triton paged path，避免把不同 harness 或 GPU clock state 的绝对时间直接比较。
- [ ] 只有 fused direct-paged 版本正确且有重复收益时，才注册正式 operator；独立
      gather spike 和中间 tensor 不进入 runtime。

Step 1 在乱序 slots、跨 physical page、非连续 K stride、非 2 次幂 head dim 和
Qwen3 `8 x 128` KV shape 上与 reference cache bitwise 一致。RTX 5090 两轮结果中，
standalone Triton write 为 `19.9-21.1 us`，reference 为 `12.5-15.4 us`，因此
端到端只有 `0.61-0.73x`。JSON 内的 profiler 数据同时显示：prefill Triton CUDA
kernel 为约 `2.68 us/write`，reference 两个 kernels 合计约 `4.27 us/write`；
decode 则约为 `0.84 us` 对 `3.16 us`。Triton device work 更少，但
Python-to-Triton dispatch 使独立调用更慢。该结果通过地址语义验收，但不构成
standalone performance win；正式路径必须在更大的 fused paged-attention 边界摊销
dispatch，或直接从当前 scheduled K/V 读取并同时保持 cache side effect。

Step 2 使用一次 `(request, context_offset, kv_head)` launch，把 request context
边界、logical position、block table lookup 和 K/V copy 全部放入 Triton。多 request
测试覆盖不同 context start/length、乱序 physical blocks、跨 page、非 2 次幂
head dim 和 Qwen3 shape，valid output 与 reference bitwise 一致。RTX 5090 两轮
512-token gather 中，prefill 标签 case 为 `2.698x/2.712x`，decode 标签 case 为
`3.103x/3.107x`；两者实际 gather 相同 K/V 数量。该 padded 输出约 2 MiB，只用于
学习和对拍；Step 3 必须把 page load 直接接到 QK、online softmax 和 PV，不能把
gather tensor 带入 runtime。

### 5.2 正式 operator

- [ ] 新增 `TritonPagedAttentionOperator` 并注册为 `"triton_paged"`。
- [ ] page write 和 fused paged read 共享当前 slot mapping/block table 语义，不复制
      Scheduler 或 KV cache policy。
- [ ] Full/SW/Chunked 的可见范围只读取 `AttentionMetadata`，不判断 spec class。
- [ ] 支持 MHA、GQA 和 MQA；由 query head 直接映射 KV head，不物化展开 K/V。
- [ ] mask 同时表达 causal upper bound 和每 query 的 context start。
- [ ] softmax/accumulation 使用与 reference 对拍后确认的稳定精度。
- [ ] mixed request batch 由一次 Triton launch 并发执行；不在 CPU 逐 request 切分或
      调用 attention。
- [ ] CPU reference 路径继续使用 `"reference"`，不要求 optimized operator 成为通用
      CPU backend。

## 6. Phase 2：parity、fallback 和回归（对应第 4 项）

### 6.1 Operator-level correctness

- [ ] full prefill：单次与 chunked prefill 对拍 reference。
- [ ] decode：读取跨多个 physical blocks 的历史并对拍 reference。
- [ ] chunked prefill：同时读取历史 cache 和当前 chunk。
- [ ] sliding window：跨 window/block boundary 的 visible positions 一致。
- [ ] chunked local：跨 aligned chunk boundary 的 visible positions 一致。
- [ ] hybrid：不同 group 共享 page buffers 时，各 layer hidden/logits 一致。
- [ ] 覆盖 MHA、GQA、MQA、不同 block size、head dimension 和 query length。
- [ ] FP16 GPU 使用冻结容差；token-level greedy 结果必须一致。

### 6.2 Engine-level correctness

- [ ] full prefill、multi-step prefill、decode 和 mixed batch 输出 token 与 reference 一致。
- [ ] prefix cache on/off、preemption/recompute、block release/reuse 不回归。
- [ ] Llama、Qwen、Mistral 和 synthetic hybrid 全部通过现有测试。
- [ ] fixed public Qwen checkpoint 在 reference/optimized 下完成同一 tokenized workload。
- [ ] `attention_operator="reference"` 是唯一 fallback 开关，关闭 optimized 后不改变
      Scheduler、cache layout 或 model weights。
- [ ] unsupported dtype/device/shape 直接报错，不在一次 request 中混用两个 operator。

### 6.3 Regression commands

- [ ] 非 GPU 单元/集成测试通过。
- [ ] attention operator GPU 定向测试通过。
- [ ] tiny model GPU eager 测试通过。
- [ ] 固定公开 checkpoint 离线回归通过。
- [ ] 完整 `python -m pytest` 通过，失败或 skip 数变化有明确解释。

## 7. Phase 3：性能对比、必要 trace 与 `torch.compile`（对应第 5 项）

### 7.1 Optimized benchmark

- [ ] 使用 Phase 0 完全相同的 model、workload、capacity、warmup 和测量次数。
- [ ] 分别保存 reference eager 与 optimized eager JSON。
- [ ] 报告 prefill throughput、decode ITL、mixed throughput、peak memory 和 KV utilization。
- [ ] 至少一个目标 workload 有稳定可复现收益；否则回到 Phase 1 review，不通过改名或
      选择性展示完成验收。
- [ ] 记录退化 workload，不用单个综合分数隐藏 prefill/decode tradeoff。

### 7.2 必要 trace 收尾

只实现 benchmark 对比和问题定位已经证明需要的 trace，不先追求完整生产级指标系统。

- [ ] 保留现有 `KVCacheLayoutTrace` 和 `KVCacheStepTrace` contract。
- [ ] 新增 frozen engine-step trace，记录 token budget、used budget、每 request
      scheduled token 数、sampling 标记和 preempted request IDs。
- [ ] 记录 request admission、resume、finish reason 和 abort transition。
- [ ] KV trace 补充 allocation/share/release 数量、prefix hit token 数和 step 后 free
      block 数；不逐 block 发出高频对象，除非诊断模式明确开启。
- [ ] trace sink 为 `None` 时不构造 event、table snapshot 或统计容器。
- [ ] benchmark 可以消费 trace 计算 preemption、prefix hit 和 recompute 等计数，但
      runtime correctness 不依赖 trace sink。

### 7.3 `torch.compile` 实验

- [ ] 只在 optimized eager 正确且有收益后开始。
- [ ] 增加一个默认关闭的显式配置开关；不自动 compile。
- [ ] compile 边界只包含 ModelRunner-owned model execution，不包含 Scheduler、Request、
      tokenizer 或 KVCacheManager。
- [ ] 测量首次 compile 成本和 steady-state；两者分别记录。
- [ ] 验证不同 batch/query shape 下的 graph count/recompile 行为。
- [ ] 与 optimized eager 做 hidden/logits/token parity。
- [ ] 若 steady-state 无稳定收益或动态 shape 导致不可接受 recompile，删除实验代码并将
      结论写入结果；V1 不保留无收益开关。

### 7.4 CPU overhead

- [ ] 用 step trace 分离 schedule、metadata preparation、model execution 和 update 时间。
- [ ] 先定位再优化；不因推测重写 Scheduler collection 或 metadata contract。
- [ ] 若 metadata 构造成为显著瓶颈，优先消除 `.tolist()`/逐 token host sync 等明确热点。
- [ ] CPU 优化必须保持 immutable SchedulerOutput 和 per-group metadata 语义。

## 8. Phase 4：V1 API/调度范围决策（对应第 6 项）

默认建议是：V1.6 聚焦 optimized attention 和可复现性能验收，不为尚无第二实现的
策略提前抽象。

- [ ] 根据实际用户场景决定 async streaming 是否进入 V1；若延期，在 requirements
      中明确为 V1.x/V2，而不是长期保留模糊未完成项。
- [ ] 若实现 streaming，先冻结取消、abort、累计/增量输出和 tokenizer decode state；
      不把同步 `LLM.generate()` 改成隐式异步。
- [ ] priority 默认延期；只有存在明确优先级语义时才提取 victim-selection policy。
- [ ] per-request max prefill chunk 只有在 benchmark 显示 TTFT/fairness 问题时实现。
- [ ] 若 streaming/priority 延期，保留当前 FIFO、running-first 和 deterministic newest
      victim 行为，不增加空 policy interface。
- [ ] 更新 `v1-core-engine.md`，将已实现、明确延期和非目标三类状态分开记录。

## 9. Benchmark 测试矩阵

| 维度 | 最小覆盖 |
|---|---|
| Operator | reference eager、optimized eager、可选 optimized compile |
| Workload | prefill、decode、mixed、cache pressure |
| Attention | full 为性能主基线；SW/chunked/hybrid 做 correctness 和代表性测量 |
| Batch | 单 request、达到 `max_num_seqs` 的 mixed batch |
| Length | 至少一个短 prompt、一个跨多 block prompt、一个长 decode |
| Cache | prefix off 基线；prefix on/reuse 代表性场景 |
| Model | Qwen3-0.6B 固定 revision；tiny models 用于快速 parity |
| Hardware | Phase 0 baseline 保留 aliyun-ecs Tesla T4；后续实验使用 RTX 5090，结果按 GPU 分开并记录实际 CUDA/PyTorch 版本 |

benchmark 结果不能只写“更快”。每份结果必须能回答：优化了哪个 workload、代价是
什么、是否改变显存/KV capacity，以及结果是否包含 compile/warmup 成本。

## 10. 预计文件变更

```text
benchmarks/benchmark_engine.py
benchmarks/results/README.md
src/micro_vllm/attention/operators/triton_paged.py
src/micro_vllm/attention/operators/registry.py
src/micro_vllm/trace.py
src/micro_vllm/core/scheduler.py
src/micro_vllm/engine/engine.py
src/micro_vllm/worker/model_runner.py
src/micro_vllm/config.py                    # 仅在 compile 实验保留时
tests/attention/operators/test_triton_paged.py
tests/integration/test_optimized_engine.py
docs/requirements/v1-core-engine.md
```

如果实现需要修改 Request 进度模型、SchedulerOutput、KVCacheLayout、model forward
签名或 per-group AttentionMetadata，应先停止并重新 review；这些不是 V1.6 性能实现的
正常要求。

## 11. 建议提交顺序

1. `bench: freeze v1 reference workloads and result schema`
2. `feat: add triton paged attention operator`
3. `test: validate optimized attention parity`
4. `bench: compare reference and optimized attention`
5. `feat: trace measured engine and cache metrics`
6. `perf: evaluate compiled model execution`（只有实验通过时保留）
7. `docs: complete v1 performance milestone`

每个提交都必须保持 reference path 可运行。benchmark、trace、operator 和 compile 不在
同一个提交中混合，方便单独验证收益和回退。

## 12. 完成标准

- [ ] benchmark workload、结果 schema、硬件和 reference baseline 可复现。
- [ ] 只有一个新增 optimized operator，现有 operator contract 不变。
- [ ] Scheduler 不感知 operator、prefill/decode kernel 或 attention spec 类型。
- [ ] optimized full/SW/chunked/hybrid 与 reference 数值和 token 对拍。
- [ ] prefix、preemption、release/reuse 和多 cache group 行为不回归。
- [ ] reference fallback 由现有配置显式选择，不存在 silent fallback。
- [ ] optimized backend 在至少一个目标 workload 上有稳定收益并完整报告 tradeoff。
- [ ] `torch.compile` 只有正确且有稳定 steady-state 收益时才保留。
- [ ] trace 关闭时不构造详细事件，benchmark 代码不进入 runtime hot path。
- [ ] streaming、priority 和 prefill chunk policy 均有明确实现或延期结论。
- [ ] V1 发布验收记录 throughput、TTFT、ITL、显存和 KV utilization。

## 13. 实现状态

2026-07-23：Phase 0 与 Phase 1 feasibility spike 已完成。Tesla T4 reference
baseline 已保存两轮原始结果；两轮 output throughput 最大差异为 0.797%，单轮
elapsed time coefficient of variation 最大为 3.92%，未观察到数量级抖动。独立
SDPA spike 在 prefill/decode 均有重复收益，实际选择 efficient attention kernel；
纯 SDPA core 无 graph break，完整 paged path 的 break 来自现有 gather 中的
`Tensor.item()`。下一步进入单一正式 operator，不先扩展 runtime trace。

2026-07-24：后续 GPU 实验迁移到 NVIDIA GeForce RTX 5090（32,607 MiB，
compute capability 12.0），并在相同 PyTorch 2.8.0+cu128、CUDA 12.8、shape、
FP16、seed、warmup 和 measurement 配置下重跑两轮 spike。prefill speedup 为
`1.340x/1.346x`，decode 为 `1.169x/1.148x`，两者仍选择 efficient attention。
Flash Attention 因显式 non-null mask 不可用；cuDNN 仅支持本次 prefill，不支持
query length 1 的 decode。鉴于相对收益较 T4 缩小，正式 operator 前先分解
write/gather/prepare/attention core 的耗时。分解结果显示 paged gather 是最大
isolated stage；下一步从 batched Triton page write 和 page-address mapping 开始，
随后实现一次 launch 并发处理 mixed requests 的 fused paged attention，不把独立
gather spike 或第一版学习 kernel 直接注册到 runtime。

2026-07-24：Triton paged path Step 1 已完成。Batched page write 使用
`grid=(total_query_tokens, num_kv_heads)`，无 request CPU loop，并与 reference
bitwise 对拍。独立 wrapper 因 dispatch 固定成本慢于 reference，但 CUDA kernel
本身减少了 device time；下一步实现仅用于地址学习和对拍的 batched page-gather
spike，随后把同一地址计算融合进 ragged paged attention。

2026-07-24：Step 2 batched page-gather 已完成。一个 kernel launch 并发处理所有
request/context offsets/KV heads，移除了 reference gather 的 CPU request loop、
`.item()` DtoH 同步和多个地址 tensor kernels；两轮 512-token context 结果稳定为
约 `2.7-3.1x`。下一步不优化或注册 padded gather，而是直接实现 fused ragged
paged attention。
