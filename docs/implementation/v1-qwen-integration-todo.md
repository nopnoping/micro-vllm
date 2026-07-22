# V1.4D Qwen3 Dense Model 与 Checkpoint 接入清单

## 1. 目标

在不改变 Scheduler、Worker 或 ModelRunner 模型生命周期边界的前提下，增加标准
`Qwen3ForCausalLM` dense 模型实现，并使用本地 tiny HF checkpoint 和固定公开
Qwen3 checkpoint 验证：

```text
local tiny or pinned Hub Qwen checkpoint
    -> existing model_loader.load_model(...)
    -> models.registry -> Qwen3Model
    -> ModelRunner warmup/profile and KV allocation
    -> Worker -> SingleProcessExecutor -> Engine
    -> greedy token IDs matching Transformers
```

本阶段证明第二个真实 dense 模型族可以复用现有控制流、Paged KV cache、reference
attention 和 ModelRunner 生命周期；Qwen 专属 config 与参数语义只进入模型/layer 层。

## 2. 范围与边界

### 本阶段实现

- [x] 标准 Hugging Face `Qwen3ForCausalLM` dense 架构。
- [x] Qwen3 的显式 head dimension、Q/K RMSNorm 和 tied embedding 参数语义。
- [x] 本地生成的 tiny Qwen3 HF-format checkpoint 参数、hidden state、logits 对拍。
- [x] loaded Qwen3 经 ModelRunner、Worker、Executor 和 Engine 生成 greedy tokens。
- [x] 固定公开 Qwen3-0.6B checkpoint 的 T4 FP16 与离线 cache 验收。

### 本阶段不实现

- Qwen2/Qwen2.5 projection bias 兼容性；本阶段只注册 Qwen3 architecture。
- Qwen-VL、多模态、MoE、量化或 remote code。
- sliding-window attention；选定模型的 `use_sliding_window` 必须为 false。
- packed QKV/gate-up checkpoint 或 tensor parallel 参数布局。
- tokenizer、chat template、字符串 prompt 和高层 `LLM.generate()`。
- temperature、top-k、top-p 或随机采样。
- 修改 Worker/ModelRunner 的职责边界；模型初始化、warmup、profile、KV 容量推导和
  physical cache allocation 继续全部由 ModelRunner 负责。

## 3. 设计决策

### 3.1 复用与专属语义

- `ModelConfig` 继续保存通用只读 HF metadata，不增加 Qwen 专属字段集合。
- `model_loader` 继续只解析来源、固定 revision、遍历 tensor 并调用模型
  `load_weights()`，不识别 Qwen parameter 名称。
- registry 只新增静态 `Qwen3ForCausalLM -> Qwen3Model` mapping。
- RMSNorm、RoPE、SwiGLU、Embedding、LM head 和 reference attention operator
  复用。
- Qwen3 attention 单独表达显式 head dimension、无 bias projections 和 per-head Q/K
  RMSNorm；不把模型判断扩散到 Scheduler、Worker 或通用 config。
- Qwen3 模型负责 `model.*` checkpoint name、tied `lm_head.weight` 和自身参数复制语义。
- ModelRunner 的 KV cache helper 从通用只读 metadata 读取显式 `head_dim`，不再假设
  `hidden_size // num_attention_heads`；warmup、容量计算和 physical allocation 的所有权
  不变。

### 3.2 选用 Qwen3-0.6B

V1.4D 直接完成 `Qwen3ForCausalLM`。相比 Qwen2.5，它要求显式 `head_dim` 和 Q/K
RMSNorm，能够验证 attention/KV cache layout 是否真正由模型 metadata 驱动，而不是
偶然依赖 Llama 的 `hidden_size // num_attention_heads`。该增量仍严格停留在模型 layer
与 ModelRunner-owned KV cache helper，不改变 Worker 的编排职责。

## 4. Phase 0：模型选择与测试隔离

### 公开模型选择

- [x] `architectures` 为标准 `Qwen3ForCausalLM`。
- [x] 公开、无 gating、无需 `trust_remote_code`。
- [x] 单文件标准 safetensors，规模适合 Tesla T4。
- [x] BF16 checkpoint 可在 T4 上以 FP16 执行。
- [x] `use_sliding_window=false`，不要求 V1.5 attention 语义。
- [x] 固定 immutable commit，并记录 dtype、参数量和文件大小。
- [x] 固定合法 prompt token IDs，不依赖 tokenizer。

选定 `Qwen/Qwen3-0.6B`，revision 固定为
`c1899de289a04d12100db370d81485cdf75e47ca`。该 revision 为 Apache-2.0 公开、
无 gating 的标准 `Qwen3ForCausalLM`；单文件 `model.safetensors` 为 BF16，包含
751,632,384 个 checkpoint tensor elements（包括单独保存的 tied LM head），文件大小
1,503,300,328 bytes；模型具有 596,049,920 个 unique parameters。config 为 28 层、
1024 hidden size、16 attention heads、8 KV heads、显式 128 head dimension、tied
embeddings、标准 RoPE，并关闭 sliding window。固定 prompt token IDs 为
`[1, 450, 7483, 310, 3444]`，Tesla T4 执行 dtype 为 FP16。

### 测试隔离

- [x] 增加离线 selected-model contract 测试。
- [x] Hub metadata 测试使用现有 `model` marker。
- [x] 未设置 `MICRO_VLLM_TEST_MODEL=1` 时外部测试统一 skip。
- [x] 普通回归不访问网络、不下载 Qwen 权重。
- [x] Hub cache 使用远端标准缓存目录，不写入仓库。

显式 metadata 验证命令：

```bash
MICRO_VLLM_TEST_MODEL=1 .venv/bin/python -m pytest -q -m model \
  tests/integration/test_hub_qwen.py
```

## 5. Phase 1：Qwen3 架构与 Config 语义

- [x] 增加 Qwen3 attention，Q/K/V/O projections 均服从 config 的无 bias 语义。
- [x] projection shape 使用显式 `head_dim`，允许 Q width 不等于 hidden size。
- [x] RoPE 前分别对每个 Q/K head 应用 head-dimension RMSNorm。
- [x] Qwen3 decoder layer 复用 pre-norm residual 和 SwiGLU 结构。
- [x] Qwen3 model 直接读取自身 HF config 字段与默认值。
- [x] KV cache shape/bytes 使用显式 `head_dim`，仍只由 ModelRunner 创建和计算。
- [x] registry 注册 `Qwen3ForCausalLM`，unsupported 错误同步列出新架构。
- [x] tiny config 初始化测试覆盖 projection shape、Q/K norm、层数和 tied identity。
- [x] 不在 Worker、ModelRunner、model_loader 中增加 Qwen 分支。

## 6. Phase 2：权重加载与本地 HF 对拍

- [x] Qwen3 `load_weights()` 映射 `model.*` 参数名和 Q/K norm 参数。
- [x] tied checkpoint 中 `lm_head.weight` 映射到 embedding parameter。
- [x] 未知参数和 shape mismatch 保留当前原生错误与 source context。
- [x] Transformers 生成确定性的 tiny Qwen3 safetensors checkpoint。
- [x] 单文件 checkpoint 通过通用 `load_model()` 加载。
- [x] loaded parameters、hidden states 和 logits 与 Transformers 对拍。
- [x] CPU FP32 和 T4 FP16 数值容差基于实测记录。

## 7. Phase 3：ModelRunner 与 Engine 集成

- [x] loaded Qwen3 只通过 initializer 进入 ModelRunner。
- [x] ModelRunner 完成 placement、warmup/profile、KV 容量推导和 physical allocation。
- [x] full prefill、强制 chunked prefill 和多 token decode 与 Transformers 对拍。
- [x] mixed batch、prefix cache、preemption/recompute 和 block reuse 语义正确。
- [x] finish、abort 和 model-length 终止后无 KV block 泄漏。

## 8. Phase 4：固定公开 Checkpoint 验收

- [x] 首次在线下载固定 revision 的 config/checkpoint。
- [x] `local_files_only=True` 从共享 cache 完全离线重载。
- [x] T4 FP16 sampling-row logits 与 Transformers eager 对拍。
- [x] full/chunked Engine greedy token IDs 逐 token 一致。
- [x] 记录加载耗时、CPU RSS、GPU peak、warmup activation 和 KV blocks。
- [x] Qwen 定向回归、全部非 model 回归和完整默认回归通过。
- [x] requirements 按实际远端证据更新，不以 metadata 测试替代真实执行验收。

## 9. 验证状态

2026-07-21 已在 `aliyun-ecs` 完成全部阶段。公开 checkpoint 首次在线加载为
130.099s，完全离线 cache 加载约 6.8s；CPU peak RSS 约 4.85GB，T4 peak allocated
约 2.515GB，warmup peak activation 为 8,931,840 bytes，可用 KV bytes 为
12,707,758,796，按显式 128 head dimension 分配 64 blocks。28 层 FP16 sampling-row
logits 最大绝对误差为 0.0390625，验收容差为 `atol=rtol=5e-2`；full/chunked
4-token greedy IDs 与 Transformers eager 完全一致。

远端回归结果：Qwen/Models/Worker 定向 `102 passed, 2 skipped`，显式非 model
`461 passed, 4 deselected`，完整默认 `461 passed, 4 skipped`；公开 Qwen3 在线验收
`2 passed, 1 deselected`，`HF_HUB_OFFLINE=1` 离线验收 `1 passed`。

## 10. 完成标准

- [x] 第二个真实 dense 模型族通过同一 ModelRunner/Engine 路径运行。
- [x] Qwen3 head dimension、Q/K norm、tied weights 等专属语义只存在于模型/layer 层。
- [x] 本地 tiny Qwen3 和公开固定 Qwen3-0.6B checkpoint 均与 Transformers 对拍。
- [x] 普通测试保持无网络，公开 checkpoint 测试保持显式启用。
- [x] 不改变 ModelRunner 对模型生命周期、warmup/profile 和 KV cache 的所有权。
- [x] `docs/requirements/v1-core-engine.md` 按最终证据更新。
