# V1.3 Full-Attention Model Execution 技术方案与执行清单

## 1. 目标

在已经完成的统一 Scheduler 和逻辑 Paged KV Cache 上，实现第一条真实模型执行链路：

```text
SchedulerOutput
    -> SingleProcessExecutor
    -> Worker
    -> ModelRunner
    -> Tiny Llama
    -> Reference Full Paged Attention
    -> Greedy Sampler
    -> ModelRunnerOutput
    -> Scheduler.update()
```

本里程碑完成后，Engine 应当能够使用随机初始化的 tiny Llama 权重生成真实 greedy token，不再依赖 FakeExecutor。

## 2. 范围决策

### 本阶段实现

- [ ] 单进程、单 Worker、单 device。
- [ ] CPU reference execution 和单 GPU eager execution。
- [ ] 单一 full-attention KV cache group。
- [ ] Worker 侧 physical KV cache tensor。
- [ ] 统一 flattened mixed batch，不划分 prefill/decode 执行路径。
- [ ] full causal reference paged attention。
- [ ] GQA/MQA head expansion。
- [ ] tiny Llama dense decoder。
- [ ] greedy sampler。
- [ ] SingleProcessExecutor。
- [ ] 与 Transformers Llama 的 layer/logits parity test。
- [ ] chunked prefill、decode、mixed batch、prefix reuse 和 recompute tests。

### 本阶段不实现

- [ ] 不实现 sliding-window attention。
- [ ] 不实现 chunked-local attention。
- [ ] 不实现多 KV cache group。
- [ ] 不实现 Mistral、Qwen 或其他模型。
- [ ] 不实现 safetensors checkpoint 加载。
- [ ] 不实现 Model Registry。
- [ ] 不实现 tokenizer 和字符串 prompt。
- [ ] 不实现 temperature、top-k、top-p sampling。
- [ ] 不实现 FlashAttention、Triton paged attention 或 FlashInfer。
- [ ] 不实现 CUDA Graph 或 `torch.compile`。
- [ ] 不实现 GPU memory profiling 和自动 block count。
- [ ] 不实现 TP、PP、DP。

## 3. 核心不变量

- [ ] Scheduler 和 KVCacheManager 只管理 block ID，不访问 tensor。
- [ ] Worker 是 model weights 和 physical KV tensors 的唯一所有者。
- [ ] ModelRunner 只消费 immutable SchedulerOutput，不修改 Request/Scheduler。
- [ ] physical block ID 必须直接对应 KV tensor 的 block dimension index。
- [ ] logical block、physical block 和 token slot 映射只有一个实现来源。
- [ ] sampled token 在下一 Scheduler step 才写入 KV cache。
- [ ] prefix hit block 只读复用，当前 step 不覆盖共享 block。
- [ ] preempt 后旧 tensor 内容无需清零；block table 与 slot overwrite 保证隔离。
- [ ] decode 和 chunked prefill 使用相同 ModelInput 和 model forward。
- [ ] optimized backend 以后必须能与本阶段 reference backend 对拍。

## 4. 目标模块关系

```text
executor/single_process.py
    -> worker/worker.py
        -> worker/model_runner.py
            -> models/llama.py
                -> layers/embedding.py
                -> layers/normalization.py
                -> layers/rotary_embedding.py
                -> layers/attention.py
                    -> layers/attention_backend.py
                -> layers/mlp.py
            -> sampling/sampler.py
        -> worker/kv_cache.py
```

依赖限制：

- [ ] `layers` 和 `models` 不导入 `worker`。
- [ ] `core` 不导入 Torch、Worker、Model 或 Layer。
- [ ] `worker/model_runner.py` 可以导入 model、sampling 和 attention metadata contract。
- [ ] attention metadata contract 放在 `layers/attention_backend.py` 或独立低层模块，避免 Layer 反向依赖 Worker。
- [ ] SingleProcessExecutor 不解释 block table、positions 或 sampling row。

## 5. Phase 0：远端环境与依赖基线

### 环境检查

- [x] 在 `aliyun-ecs` 记录 Python 版本。
- [x] 在 `aliyun-ecs` 记录 PyTorch、CUDA、Transformers 版本。
- [x] 确认远端至少有一块可见 CUDA GPU。
- [x] 记录 GPU 型号、compute capability 和支持 dtype。
- [x] 确认现有 CPU-only test suite 在远端通过。
- [x] 不在本地机器运行代码或测试。

### `pyproject.toml`

- [x] 增加经过远端验证的 PyTorch runtime dependency/兼容说明。
- [x] 增加 Transformers test dependency，用于 parity test。
- [x] 不提前添加 safetensors、flash-attn、triton 等本阶段未使用依赖。
- [x] 配置 `gpu` pytest marker。
- [x] CPU reference tests 和 GPU tests 可以独立选择。
- [x] 稳定测试命令保留在本清单，不写入项目根 `AGENTS.md`。

### 同步与执行规则

所有编辑只发生在本地仓库。每次运行代码或测试前，从 `micro-vllm` 仓库根目录执行：

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/
```

测试只能通过远端执行，例如：

```bash
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m "not gpu"'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m gpu'
```

- [x] 不在远端直接编辑源文件。
- [x] 每次远端测试前重新同步本地 working tree。
- [x] 远端生成的源码修改不得反向覆盖本地。
- [x] 测试报告记录远端 Python、Torch、CUDA 和 GPU 信息。

### Phase 0 远端验证记录（2026-07-19）

- Host：`aliyun-ecs`，项目环境：`/root/micro-vllm/.venv`。
- Python：`3.12.3`。
- PyTorch：`2.8.0+cu128`，CUDA runtime build：`12.8`。
- Transformers：`4.57.6`。
- GPU：1 x Tesla T4，compute capability `7.5`，显存
  `15,636,037,632` bytes，driver `580.126.09`。
- dtype：FP32 和 FP16 CUDA matmul 实测通过；
  `torch.cuda.is_bf16_supported()` 返回 `True`，但首版 T4 GPU 数值测试选择
  FP16，不假定原生 BF16 Tensor Core 支持。
- CPU-only：`177 passed, 1 deselected`。
- GPU-only：`1 passed, 177 deselected`。
- 完整测试：`178 passed`。
- 项目未直接声明 safetensors、flash-attn 或 triton；Transformers/PyTorch
  安装产生的传递依赖不作为本阶段功能使用。

## 6. Phase 1：模型长度和执行配置

### Scheduler/model length

- [ ] 为 Scheduler 可访问的配置增加 `max_model_len`。
- [ ] 校验 `max_model_len > 0`。
- [ ] 首版要求 `prompt_length < max_model_len`，至少保留一个生成 token 位置。
- [ ] 调度不得产生超出 `[0, max_model_len)` 的 model input position。
- [ ] 达到 max model length 时以明确 finish reason 结束请求。
- [ ] max tokens 和 max model length 同时达到时终止顺序有明确测试。
- [ ] cache block capacity 不代替 max model length 语义。

### Llama architecture config

- [ ] 定义最小、frozen `LlamaModelConfig` 或等价结构。
- [ ] 包含 `vocab_size`。
- [ ] 包含 `hidden_size`。
- [ ] 包含 `intermediate_size`。
- [ ] 包含 `num_hidden_layers`。
- [ ] 包含 `num_attention_heads`。
- [ ] 包含 `num_key_value_heads`。
- [ ] 包含 `max_position_embeddings`。
- [ ] 包含 `rms_norm_eps`。
- [ ] 包含 `rope_theta`。
- [ ] 包含 activation 和是否 tied embedding 所需的最小字段。
- [ ] 校验 head divisibility、hidden/head dim 和 GQA group ratio。
- [ ] 不创建覆盖所有未来模型的万能 config。

### Runner/device config

- [ ] device 和 dtype 显式传入 Worker/ModelRunner。
- [ ] CPU reference 默认使用 float32。
- [ ] GPU test 至少验证一种 FP32/BF16/FP16 路径，依据远端硬件选择。
- [ ] 不使用 mutable global device context。

## 7. Phase 2：Physical KV Cache

### 新增 `src/micro_vllm/worker/kv_cache.py`

建议每层使用以下布局：

```text
[2, num_blocks, block_size, num_kv_heads, head_dim]
 │
 ├── 0 = key
 └── 1 = value
```

### 数据与所有权

- [ ] 定义只在 Worker 数据面使用的 physical KV cache container/helper。
- [ ] 为每个 decoder layer 创建独立 cache tensor。
- [ ] tensor 数量等于 `num_hidden_layers`。
- [ ] block dimension 等于逻辑 CacheConfig 的 `num_blocks`。
- [ ] block size 等于逻辑 CacheConfig 的 `block_size`。
- [ ] KV head 数和 head dim 来自 Llama model config。
- [ ] tensor dtype/device 与 Worker 配置一致。
- [ ] physical cache 不保存 Request 或 block table。
- [ ] 不在初始化时依赖真实权重文件。

### API

- [ ] 提供按 layer index 获取 cache tensor 的只读容器接口。
- [ ] 不向 control plane 暴露 tensor。
- [ ] 提供 shape/dtype/device validation。
- [ ] 提供显式 release/shutdown 所需操作。
- [ ] 不自动清零被逻辑层释放的 block。

### 测试

- [ ] layer 数、tensor shape、dtype 和 device 正确。
- [ ] physical block ID 对应 block dimension index。
- [ ] 不同 layer tensor 不共享 storage。
- [ ] CPU 和 GPU allocation 测试分离。
- [ ] invalid model/cache shape 在模型执行前失败。

## 8. Phase 3：Attention Metadata 与 ModelInput

### `AttentionMetadata`

放置在 `layers/attention_backend.py` 或独立低层 schema module。

- [ ] frozen dataclass，不包含 Request 对象。
- [ ] 包含 flattened query positions。
- [ ] 包含 request query start locations，长度为 `num_requests + 1`。
- [ ] 包含每个 request 计算后的 context length。
- [ ] 包含 padded physical block table tensor。
- [ ] 包含 flattened slot mapping。
- [ ] 包含 block size。
- [ ] padding block ID 使用明确 sentinel，例如 `-1`。
- [ ] metadata shape 和 request 数具有一致性校验。
- [ ] backend 可以仅凭 metadata 和 layer KV cache 完成读写。

### `ModelInput`

定义在 `worker/model_runner.py`，仅供 Runner 内部使用。

- [ ] 包含 flattened `input_ids`。
- [ ] 包含 flattened absolute `positions`。
- [ ] 包含 `AttentionMetadata`。
- [ ] 包含 request IDs，顺序与 SchedulerOutput 一致。
- [ ] 包含需要采样的 flattened row indices。
- [ ] 不包含可变 Request。
- [ ] 不包含 SamplingParams；本阶段仅支持 greedy。

### Flattening

- [ ] 按 SchedulerOutput request 顺序拼接 input token IDs。
- [ ] position 从每个 ScheduledRequest 的 `num_computed_tokens` 开始。
- [ ] query start locations 以零开始并以总 query token 数结束。
- [ ] mixed batch 中不同 request positions 相互独立。
- [ ] 每个 scheduled token 都有且只有一个 slot mapping。
- [ ] `requires_sampling=True` 时采样 row 是该 request 的最后一个 query row。
- [ ] `requires_sampling=False` 的 request 不产生 sampling row。

### Slot mapping

唯一公式：

```python
logical_block = position // block_size
block_offset = position % block_size
physical_block = block_table[logical_block]
slot = physical_block * block_size + block_offset
```

- [ ] logical block 必须存在且不能为 `None`。
- [ ] physical block ID 必须在 `[0, num_blocks)`。
- [ ] slot 必须在 physical cache flattened capacity 内。
- [ ] 公式只在一个 helper 中实现，ModelRunner 和 backend 不重复实现。

### Block table tensor

- [ ] 将不同长度 block tables pad 到 batch 最大长度。
- [ ] padding 不被 attention gather 使用。
- [ ] tensor dtype 使用足够容纳 block ID 的整数类型。
- [ ] tensor 与 ModelInput 位于同一执行 device。

### 测试

- [ ] 单 request prefill metadata。
- [ ] 单 request decode metadata。
- [ ] chunked prefill 的非零起始 position。
- [ ] mixed decode=1/prefill=N batch flattening。
- [ ] 多 request 不同 block table 长度 padding。
- [ ] block boundary 前后 slot mapping。
- [ ] prefix hit 后从非零 computed position 开始。
- [ ] invalid/missing/None block mapping 被拒绝。
- [ ] sampling row indices 与 request ID 顺序一致。

## 9. Phase 4：Reference Full Paged Attention

### Backend contract

- [ ] 定义最小 `AttentionBackend` protocol 或明确 callable contract。
- [ ] contract 接受 Q、K、V、单层 physical KV cache 和 AttentionMetadata。
- [ ] contract 返回与 Q 相同 flattened query 数的 attention output。
- [ ] backend 不访问 SchedulerOutput、Request 或 KVCacheManager。
- [ ] backend 不执行 QKV projection、RoPE 或 output projection。

### KV write

- [ ] 根据 slot mapping 将当前 token K/V 写入 physical cache。
- [ ] 写入前 reshape cache 为可按 flattened slot 索引的视图。
- [ ] 每个 scheduled token 写入唯一 slot。
- [ ] prefix/shared slots 不在当前 query slot mapping 中被覆盖。
- [ ] 同一 plan 内较早 query 的 K/V 对后续 query 可见。

### KV gather

- [ ] 对每个 request 独立处理 query range。
- [ ] 对绝对 query position `p` 读取 token positions `[0, p]`。
- [ ] 使用 request block table 将 logical positions 映射到 physical cache。
- [ ] 不读取 padding block IDs 或 context length 之外的 stale values。
- [ ] chunked prefill 同时读取历史 cache 和当前 chunk 已写入的 K/V。
- [ ] decode 读取全部历史 full-attention context。

### Attention math

- [ ] 应用 `1 / sqrt(head_dim)` scaling。
- [ ] 使用 causal visibility，不允许 query 看到未来 token。
- [ ] softmax 计算使用数值稳定 dtype，必要时临时转换 float32。
- [ ] 输出转换回输入/model dtype。
- [ ] 支持 `num_attention_heads == num_key_value_heads`。
- [ ] 支持 GQA/MQA，通过确定性 repeat/interleave 扩展 KV heads。
- [ ] 校验 attention heads 可被 KV heads 整除。
- [ ] reference 实现优先可读性，允许 request/query Python loop。

### Reference tests

- [ ] 单 token self-attention 与 dense reference 对拍。
- [ ] 完整 causal prefill 与 dense masked attention 对拍。
- [ ] chunked prefill 与一次性 prefill 对拍。
- [ ] prefill 后逐 token decode 与完整序列 attention 对拍。
- [ ] mixed batch 与每 request 独立执行对拍。
- [ ] GQA 和 MQA 与显式 repeat KV dense reference 对拍。
- [ ] block table 使用非连续 physical block ID 时结果不变。
- [ ] block ID 释放后复用不会读取新 context 之外的 stale KV。
- [ ] prefix shared block 在两个 request 中产生一致历史 attention。
- [ ] FP32 CPU 使用严格容差；GPU low-precision 使用记录的合理容差。

## 10. Phase 5：Llama 基础 Layers

### `layers/normalization.py`

- [ ] 实现 Llama RMSNorm。
- [ ] variance accumulation 使用稳定 dtype。
- [ ] parameter shape、epsilon 和输出 dtype 正确。
- [ ] 与 Transformers RMSNorm 对拍。

### `layers/rotary_embedding.py`

- [ ] 实现标准 Llama RoPE。
- [ ] 接受 flattened absolute positions。
- [ ] 只旋转 Q/K，不旋转 V。
- [ ] 支持动态计算或足够长度的 cos/sin cache。
- [ ] 不实现 rope scaling 变体。
- [ ] 与 Transformers 标准 RoPE 对拍。

### `layers/linear.py`

- [ ] 本阶段使用普通单卡 dense linear。
- [ ] 不提前加入 TP shard 或 collective。
- [ ] bias 行为与 Llama 对齐。
- [ ] 保留清晰 parameter name，方便下一里程碑权重加载。

### `layers/mlp.py`

- [ ] 实现 gate projection。
- [ ] 实现 up projection。
- [ ] 实现 SiLU gate activation。
- [ ] 实现 element-wise gate/up multiply。
- [ ] 实现 down projection。
- [ ] 与 Transformers Llama MLP 对拍。

### `layers/embedding.py`

- [ ] 实现 token embedding。
- [ ] 实现 LM head。
- [ ] tied/untied embedding 行为由 config 明确控制。
- [ ] 本阶段不实现 vocab parallel。

## 11. Phase 6：Llama Attention 和 Decoder

### `layers/attention.py`

- [ ] 实现 q_proj、k_proj、v_proj。
- [ ] 支持独立 attention heads 和 KV heads。
- [ ] reshape 为 flattened token/head/head_dim 表示。
- [ ] 对 Q/K 应用 RoPE。
- [ ] 调用注入的 AttentionBackend。
- [ ] 合并 heads 并执行 o_proj。
- [ ] 不根据 prefill/decode 选择不同 forward。
- [ ] 不读取模型类型以外的 Scheduler 状态。

### `models/llama.py`

- [ ] 实现 DecoderLayer pre-norm residual 结构。
- [ ] attention residual 路径与 Transformers 对齐。
- [ ] MLP residual 路径与 Transformers 对齐。
- [ ] 实现 token embedding。
- [ ] 实现 N 层 decoder iteration。
- [ ] 每层接收对应 physical KV cache tensor。
- [ ] 实现 final RMSNorm。
- [ ] `forward()` 返回所有 scheduled query hidden states。
- [ ] `compute_logits()` 与 forward 分离。
- [ ] LM head 只在 ModelRunner 指定 sampling rows 上调用或明确先 gather hidden states。
- [ ] model 不执行 sampling。
- [ ] model 不加载 checkpoint 文件。

### Tiny config

用于快速测试的默认 tiny config 建议：

```text
vocab_size = 128
hidden_size = 64
intermediate_size = 128
num_hidden_layers = 2
num_attention_heads = 4
num_key_value_heads = 2
max_position_embeddings = 128
```

- [ ] 固定 random seed。
- [ ] attention dropout 和 hidden dropout 关闭。
- [ ] model 强制 eval mode 执行。
- [ ] CPU test 使用小 batch、短 context，保持快速。

## 12. Phase 7：Greedy Sampler

### `sampling/sampler.py`

- [ ] 接受 sampling hidden states 或 logits。
- [ ] 只处理 ModelRunner 指定的 sampling rows。
- [ ] 使用 `argmax(dim=-1)` 选择 token。
- [ ] 输出 token 顺序与 sampling request IDs 完全一致。
- [ ] 生成 `SampledToken(request_id, token_id)`。
- [ ] 没有 sampling row 时返回空 ModelRunnerOutput。
- [ ] 不读取或修改 Request。
- [ ] 不实现随机数状态。

### 测试

- [ ] 单 row greedy token 正确。
- [ ] 多 row 映射到正确 request ID。
- [ ] 空 sampling batch 正确返回。
- [ ] 相同最大值 tie 行为与 PyTorch argmax 一致。
- [ ] CPU/GPU 输出 token ID 一致。

## 13. Phase 8：ModelRunner

### `worker/model_runner.py`

- [ ] ModelRunner 构造时接收 model、physical KV cache、device、dtype 和 block size。
- [ ] 将 SchedulerOutput 转换为 ModelInput。
- [ ] 将输入 metadata tensor 移到正确 device。
- [ ] 在 inference mode 下执行 model。
- [ ] 只 gather sampling row hidden states。
- [ ] 调用 model `compute_logits()`。
- [ ] 调用 GreedySampler。
- [ ] 返回 ModelRunnerOutput。
- [ ] 保持 SchedulerOutput request 顺序。
- [ ] 不修改 SchedulerOutput 或其嵌套 tuple。
- [ ] 不缓存 Request Python 对象。

### Batch 行为

- [ ] 支持 batch 中所有 request 都不采样。
- [ ] 支持部分 request 采样、部分继续 prefill。
- [ ] 支持全部 request 采样。
- [ ] 支持 request 具有不同 scheduled token 数。
- [ ] 支持 non-contiguous physical blocks。
- [ ] 支持 prefix hit 后的非零 position 起点。

### 测试

- [ ] metadata builder 与预期 tensors 对拍。
- [ ] Fake/Tiny model 验证 sampling row 选择。
- [ ] ModelRunner 输出 request ID 不依赖隐式 batch index。
- [ ] ModelRunner 不改变 plan。
- [ ] 同一 pending plan 重试执行结果一致。

## 14. Phase 9：Worker 和 SingleProcessExecutor

### `worker/worker.py`

- [ ] Worker 构造时显式接收 model/cache/device 配置。
- [ ] Worker 创建并拥有 model。
- [ ] Worker 创建并拥有 physical KV cache。
- [ ] Worker 创建并拥有 ModelRunner。
- [ ] model 移动到目标 device/dtype 并进入 eval mode。
- [ ] `execute(plan)` 委托 ModelRunner。
- [ ] 提供显式 shutdown，释放 model/cache references。
- [ ] CUDA shutdown 行为不隐藏全局 device side effect。

### `executor/single_process.py`

- [ ] 实现 Executor protocol。
- [ ] 构造时接收或创建一个 Worker。
- [ ] `execute(plan)` 只委托 Worker。
- [ ] 不重排 request。
- [ ] 不修改 plan。
- [ ] 不做 admission、preemption 或 block allocation。
- [ ] 提供 shutdown 委托。

### Executor protocol

- [ ] 评估是否将 shutdown 加入协议。
- [ ] 若加入，更新 FakeExecutor 和现有 contract tests。
- [ ] 不加入尚未使用的 distributed RPC、memory profiling 等方法。

### 测试

- [ ] Executor 原样传递 plan。
- [ ] Worker/Runner 的 ModelRunnerOutput 原样返回。
- [ ] shutdown 后重复 execute 行为明确。
- [ ] CPU Worker integration test。
- [ ] GPU Worker integration test 使用 pytest marker。

## 15. Phase 10：Transformers Parity

### 测试权重准备

- [ ] 创建相同 tiny architecture 的 Transformers Llama model。
- [ ] 使用固定 seed 初始化。
- [ ] 将 Transformers state dict 直接映射/复制到 micro-vLLM model。
- [ ] 该测试不读取 safetensors 文件。
- [ ] 明确记录 parameter naming difference。

### 分层 parity

- [ ] RMSNorm 输出对拍。
- [ ] RoPE Q/K 输出对拍。
- [ ] MLP 输出对拍。
- [ ] Attention 单层输出对拍。
- [ ] DecoderLayer 输出对拍。
- [ ] 完整 model hidden states 对拍。
- [ ] 完整 logits 对拍。
- [ ] greedy token 完全一致。

### 执行模式 parity

- [ ] 一次性 full prefill 与 Transformers logits 对拍。
- [ ] chunked prefill 最后 logits 与 full prefill 对拍。
- [ ] prefill + one-token decode 与完整序列 forward 对拍。
- [ ] 多次 decode 与完整序列最后 logits 对拍。
- [ ] mixed batch 与逐 request 独立执行对拍。
- [ ] GQA config 对拍。
- [ ] CPU FP32 使用严格容差。
- [ ] GPU low-precision 容差独立记录。

## 16. Phase 11：Engine 端到端测试

- [ ] Engine 使用真实 SingleProcessExecutor。
- [ ] 单 request 能从 WAITING 运行到 FINISHED。
- [ ] 生成 token 来自 tiny Llama logits，不使用 FakeExecutor。
- [ ] chunked prefill 与 full prefill 生成相同首 token。
- [ ] mixed decode/prefill batch 与独立执行生成一致。
- [ ] prefix cache on/off 生成一致。
- [ ] prefix hit 后 physical shared KV 被正确读取。
- [ ] preemption/recompute 与无抢占生成一致。
- [ ] physical block 被其他 request 复用后无 stale-value 污染。
- [ ] EOS/max tokens/max model length 正确结束。
- [ ] abort 后逻辑 block 释放，Worker 不再访问该 request block table。
- [ ] FakeExecutor control-loop tests 保留为快速 core regression tests。

## 17. 可观察性

- [ ] ModelRunner 可报告 flattened token 数。
- [ ] 可报告 request 数和 sampling row 数。
- [ ] 可报告 physical KV cache shape、dtype 和总字节数。
- [ ] 可报告 prefill/decode token 构成，但不创建两条执行路径。
- [ ] reference attention 测试失败时能关联 request ID、position 和 block table。
- [ ] 默认不记录 logits、hidden states 或完整 prompt token 内容。

## 18. 推荐提交顺序

1. [ ] `build: add model-execution dependencies and remote test markers`
2. [ ] `feat: add model length and tiny Llama configuration`
3. [ ] `feat: add worker-owned physical KV cache`
4. [ ] `feat: build flattened model inputs and attention metadata`
5. [ ] `feat: implement reference full paged attention`
6. [ ] `feat: implement Llama layers and tiny causal model`
7. [ ] `feat: add greedy sampler and model runner`
8. [ ] `feat: add worker and single-process executor`
9. [ ] `test: add Transformers and execution-mode parity coverage`
10. [ ] `test: run real model execution through the engine`
11. [ ] `docs: mark completed V1 model-execution requirements`

每个提交必须保持远端 CPU test suite 可运行；GPU 相关提交同时运行对应 GPU marker tests。

## 19. 最终验收

- [ ] 全部编辑来自本地 working tree。
- [ ] 最终测试前已同步到 `/root/micro-vllm`。
- [ ] 远端 CPU-only test suite 通过。
- [ ] 远端 GPU test suite 通过。
- [ ] tiny Llama logits 与 Transformers 在规定容差内一致。
- [ ] greedy token 与 Transformers 完全一致。
- [ ] full prefill、chunked prefill 和 decode 结果一致。
- [ ] mixed batch 与独立执行结果一致。
- [ ] prefix cache on/off 结果一致。
- [ ] preempt/recompute 与无抢占结果一致。
- [ ] physical block reuse 无 stale KV 污染。
- [ ] Engine 使用真实 Executor 完成生成。
- [ ] Scheduler、KVCacheManager 和 core modules 不导入 Torch。
- [ ] 生产代码中不存在 prefill/decode 双执行状态机。
- [ ] 本阶段未引入多 cache group 或多 attention 类型。
- [ ] 更新并勾选实际完成的 V1 requirements。
- [ ] 工作区只包含本里程碑相关变更。
