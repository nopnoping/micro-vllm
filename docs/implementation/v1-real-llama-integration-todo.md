# V1.4C 真实 Llama Checkpoint 对拍与 Engine 集成技术方案

## 1. 目标

将已经实现的 Model Loader 接入真实执行链路，证明从 Hugging Face checkpoint 到
Engine greedy token 的完整路径正确：

```text
local path or pinned HF repo
    -> delayed model initializer
    -> Worker process/device orchestration
    -> ModelRunner -> model_loader.load_model(...) -> loaded LlamaModel
    -> warmup/profile -> derived block capacity
    -> ModelRunner-owned Physical KV Cache
    -> SingleProcessExecutor
    -> Engine
    -> greedy token IDs
```

本阶段的重点不是继续扩展 Loader，而是验证真实 checkpoint 的参数、logits、token 和
Paged KV 执行语义，并消除 Worker 原先重新构造随机模型的断点。

## 2. 当前基础与验证状态

当前代码已经具备：

- 本地目录和 Hugging Face Hub 模型解析；
- commit-pinned config/checkpoint 下载；
- 单文件和多 shard safetensors 流式读取；
- 静态 architecture registry；
- `LlamaModel.load_weights()` 参数复制和 tied embedding 映射；
- tiny Llama 与 Transformers 的 forward/logits 数值对拍；
- 随机初始化 Llama 的 Engine、chunked prefill、prefix cache 和 preemption 测试。

2026-07-21 已在 `aliyun-ecs` 补完 V1.4B 的 Model Loader、CPU/GPU 数值对拍和完整
回归；本清单只按实际远端证据勾选完成项。

## 3. 范围

### 本阶段实现

- [x] 使用 Transformers 生成标准 HF 格式 tiny Llama checkpoint 并通过
      `load_model()` 加载。
- [x] 校验加载后的参数、hidden states 和 logits 与 Transformers 一致。
- [x] ModelRunner 调用延迟 initializer 构造或加载模型，并持有模型生命周期。
- [x] 使用 loaded model 打通 Worker、Executor 和 Engine。
- [ ] 真实 checkpoint 的 full/chunked prefill、decode 和 greedy token 与
      Transformers 对拍。
- [x] 使用一个公开、固定 commit 的小型 Llama-family checkpoint 完成远端 smoke test。
- [x] 在 loaded model 上验证 prefix cache、preemption/recompute 和 block reuse。

### 本阶段不实现

- tokenizer 或字符串 prompt。
- `LLM.generate()` 高层离线 API。
- temperature、top-k、top-p 或随机采样。
- Qwen、Mistral 或其他模型。
- packed QKV/gate-up 参数布局。
- meta-device、分片初始化或生产级低内存加载。
- Hub 登录流程、模型发布或 checkpoint 转换工具。
- TP、PP、DP、量化或 optimized attention。

## 4. 核心设计决策

### 4.1 ModelRunner 负责模型生命周期

Phase 3 调整前 Worker 接收 `ModelConfig`，再通过 Registry 构造一个随机模型；这会丢失
`load_model()` 已经加载的参数。调整后的边界使用延迟 initializer，但只允许
ModelRunner 调用它：

```python
model_initializer = lambda: load_model(config)
worker = Worker(
    model_initializer,
    cache_initialization_config,
    scheduler_config,
    execution_config,
)
cache_config = worker.cache_config  # actual profiled capacity for control plane
```

- [x] Worker 不再调用 models registry。
- [x] Worker 不识别 Llama/Qwen architecture。
- [x] Worker 只把 initializer 交给 ModelRunner，不直接调用或保留 model。
- [x] ModelRunner 调用 initializer，校验模型 contract，并持有 model。
- [x] ModelRunner 负责模型 device/dtype placement 和 eval mode。
- [x] ModelRunner 在最终 KV cache 分配前执行 warmup/profile。
- [x] ModelRunner 计算 available GPU KV bytes 和 bytes per block，推导实际 block 数。
- [x] `initialize_kv_cache()` 不接收外部分配的 cache 或最终 block 数。
- [x] ModelRunner 创建并拥有 Physical KV tensor storage。
- [x] Worker 向 control plane 暴露 runner 返回的实际 `CacheConfig`。
- [x] shutdown 释放 ModelRunner 的 model/cache tensor 引用，
      不修改 checkpoint/cache 文件。
- [x] 不同时保留“config 构造”“model 实例注入”和“initializer”多套 Worker 主路径。

如果静态类型需要公共模型契约，只定义 ModelRunner 实际消费的窄 protocol：
`config`、model call 和 `compute_logits()`；不建立通用模型基类。

warmup、GPU memory profiling、模型参数/激活内存计算以及由此产生的可用 KV 容量
推导都属于 ModelRunner 生命周期。CUDA 路径按 `gpu_memory_utilization` 计算预算并扣除
non-KV memory 和 peak activation；CPU reference path 因无 GPU profile，使用显式上限
作为确定性容量。

### 4.2 Loader 与执行设备分离

- [x] `load_model()` 继续返回 CPU 默认 dtype、eval mode 模型。
- [x] checkpoint source 继续只产生 CPU tensor。
- [x] ModelRunner 是模型 device/dtype placement 的唯一位置。
- [x] Model Loader 不创建 KV cache，不读取 SchedulerConfig。
- [x] 通过 ModelRunner warmup/profile 记录目标 device 峰值激活和可用 KV memory；
      不引入 meta loading。

### 4.3 两级 checkpoint 验证

使用两类 checkpoint，分别解决可重复性和真实兼容性：

1. **本地生成的 HF-format tiny checkpoint**
   - 由 Transformers `LlamaForCausalLM.save_pretrained(...,
     safe_serialization=True)` 生成；
   - 无网络、尺寸很小，进入常规测试；
   - 固定 seed，验证完整格式、参数名称和数值对拍。
2. **公开 Hub checkpoint**
   - 选择公开、无需 remote code、标准 `LlamaForCausalLM`、safetensors 的小模型；
   - 固定 immutable commit SHA；
   - 作为显式远端模型测试，不进入普通快速测试的强制下载路径。

本地生成 checkpoint 是 Loader 正确性的自动化基线；公开 checkpoint 才满足真实外部
模型兼容性验收，两者不能互相替代。

## 5. Phase 0：模型选择与测试隔离

### 公开模型选择标准

- [x] `architectures` 包含 `LlamaForCausalLM`。
- [x] 权重为标准单文件或 sharded safetensors。
- [x] 不要求 `trust_remote_code`。
- [x] 不要求鉴权或接受额外许可证流程。
- [x] config 不依赖当前未实现的 RoPE scaling、sliding window 或 bias 结构。
- [x] 参数规模适合 Tesla T4 和远端可用磁盘/内存。
- [x] 记录 repo id、commit SHA、checkpoint dtype、参数量和下载大小。
- [x] 固定一组合法 prompt token IDs，不依赖 tokenizer。

选定 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`，revision 固定为
`d9128824c0c80111be21424e68086f52413fb413`。该 revision 为 Apache-2.0 公开、
无 gating 的标准 `LlamaForCausalLM`；单文件 `model.safetensors` 为 BF16，
包含 1,100,048,384 个参数，文件大小 2,200,119,864 bytes。config 为
22 层、2048 hidden size、32 attention heads 和 4 KV heads，没有 RoPE scaling、
sliding window、attention bias 或 remote-code 配置。checkpoint 保持 BF16 记录；
Tesla T4 执行使用 FP16。固定 prompt token IDs 为
`[1, 450, 7483, 310, 3444]`。

### 测试分类

- [x] 增加 `model` 或等价 pytest marker，表示需要外部 checkpoint/cache。
- [x] 普通 unit/integration 测试不访问网络。
- [x] 公开模型测试在未显式提供开关时 skip，而不是自动下载。
- [x] 显式模型测试命令记录 repo、revision、device 和 dtype。
- [x] Hub cache 位于远端标准缓存目录，不写入仓库。

`tests/integration/test_hub_llama.py` 的外部元数据校验使用 `model` marker。
除非 `MICRO_VLLM_TEST_MODEL=1`，否则该类测试由 collection hook 统一
skip。测试不传 `cache_dir`，因此 Hugging Face 文件位于执行用户的标准
Hub cache，不会写入仓库。显式命令为：

```bash
MICRO_VLLM_TEST_MODEL=1 .venv/bin/python -m pytest -q -m model \
  tests/integration/test_hub_llama.py
```

该命令使用测试文件固定的 repo/revision；Phase 5 执行配置固定为
`device=cuda` 和 `dtype=float16`。

2026-07-21 Phase 0 远端验证：默认不开启外部模型时为
`1 passed, 1 skipped in 0.01s`。阿里云 DNS 将 `huggingface.co` 解析到错误 IP，
官方端点在 60 秒内超时；临时设置 `HF_ENDPOINT=https://hf-mirror.com`
后显式元数据测试为 `1 passed, 1 deselected in 4.14s`。

## 6. Phase 1：HF-format Tiny Checkpoint Fixture

- [x] 使用 Transformers tiny `LlamaConfig` 构造 reference model。
- [x] 固定 torch seed 并设置 eval mode。
- [x] 使用 `save_pretrained(..., safe_serialization=True)` 写入临时目录。
- [x] 保存的 `config.json` 包含受支持 architecture。
- [x] checkpoint 使用标准 `model.*` 和 `lm_head.weight` 名称。
- [x] fixture 默认生成单文件 checkpoint。
- [x] 增加一个强制小 shard size 的多 shard fixture，覆盖 index 路径。
- [x] fixture 不使用网络，不写入仓库，不依赖进程外缓存。

`tests/conftest.py` 提供 `tiny_hf_llama_checkpoint` 和
`sharded_tiny_hf_llama_checkpoint`。两者使用相同的 tiny GQA config 和
seed 0，并使用 `torch.random.fork_rng()` 避免污染其他测试的 RNG 状态。
分片 fixture 使用 `max_shard_size="5KB"`；所有输出只位于 pytest
`tmp_path`。

2026-07-21 Phase 1 远端验证：`tests/integration/test_loaded_llama.py` 为
`3 passed in 0.36s`。同时修复 V1.4B 留下的 Loader 缺失字段错误语义和
Worker 测试对旧 config 属性的引用，定向测试为 `8 passed in 1.10s`，
完整默认测试为 `406 passed, 1 skipped in 2.24s`。

## 7. Phase 2：加载参数与数值对拍

### 参数

- [x] `load_model()` 加载单文件 tiny checkpoint。
- [x] `load_model()` 加载多 shard tiny checkpoint。
- [x] micro-vLLM 与 Transformers 的参数名称映射集合一致。
- [x] 每个 untied parameter 数值一致。
- [x] tied embedding 保持对象 identity 和数值一致。
- [x] 加载后 model 为 eval mode、CPU 默认 dtype。
- [x] 两次加载返回互不共享参数的模型实例。

### Forward 和 logits

- [x] 使用相同 token IDs 和 positions。
- [x] 为 micro-vLLM 创建等价 physical KV cache 和 metadata。
- [x] final hidden states 与 Transformers 对拍。
- [x] logits 与 Transformers 对拍。
- [x] CPU FP32 使用明确的 `atol`/`rtol`。
- [x] GPU FP16 使用单独、较宽且有依据的容差。
- [x] 失败报告包含 checkpoint、dtype、shape 和最大误差。

`tests/integration/test_loaded_llama.py` 对单文件和分片 checkpoint 的完整
state dict 做集合和逐 tensor 精确比较，并增加 tied checkpoint fixture 验证
embedding/LM head identity。数值对拍使用同一组 `[1, 3, 5, 7]` token IDs、
positions 和非连续 physical block table。CPU FP32 容差为 `atol=rtol=1e-5`；
Tesla T4 FP16 容差为 `atol=rtol=3e-3`，与已有随机 tiny Llama 对拍一致。

2026-07-21 Phase 2 远端验证：定向测试为 `9 passed in 1.31s`，其中
GPU FP16 用例在 Tesla T4 上实际运行；完整默认测试为
`412 passed, 1 skipped in 2.33s`。

## 8. Phase 3：ModelRunner 模型生命周期

- [x] Worker 构造参数从 `ModelConfig` 改为延迟 `ModelInitializer`。
- [x] Worker 不再导入 `get_model_class`。
- [x] Worker 不直接调用 initializer，也不持有 model 引用。
- [x] ModelRunner 调用 initializer 并持有返回的 model。
- [x] model 缺少执行所需 contract 时由 ModelRunner 在初始化阶段明确失败。
- [x] ModelRunner 将 loaded CPU model 移动到配置 device/dtype 并设为 eval mode。
- [x] ModelRunner 在 `initialize_kv_cache()` 内运行 warmup/profile。
- [x] CUDA 路径根据 GPU memory utilization、non-KV memory 和 peak activation
      计算 available KV bytes。
- [x] ModelRunner 根据模型结构、block size 和 dtype 计算 bytes per block 和实际容量。
- [x] ModelRunner 创建并拥有 physical KV tensor storage。
- [x] Worker 暴露实际 `CacheConfig`，control plane 使用同一容量创建逻辑 block pool。
- [x] shutdown 行为保持幂等，并由 ModelRunner 清除 model/cache/tensor 引用。
- [x] 更新现有 Worker、Executor 和 Engine fixtures，显式提供延迟 initializer。
- [x] 随机 tiny model 测试继续保留，作为快速执行基线。
- [x] CPU reference path 使用 `max_num_blocks` 作为确定性容量；CUDA 只将其作为可选上限。

Worker 只接受延迟 `ModelInitializer` 并立即转交 ModelRunner；只有 ModelRunner 会调用
initializer，校验 `ModelConfig`、forward 和 `compute_logits()` contract，执行模型
placement，并持有最终模型。placement 不设置备用 device/dtype，失败原样向上传播。
ModelRunner 的无参数 `initialize_kv_cache()` 先使用临时 cache 跑 warmup，CUDA 路径测量
peak activation 并读取 GPU free/total memory，再计算并分配最终 Physical KV Cache。
Worker 不持有 physical tensor，只保存并暴露 runner 返回的实际 `CacheConfig`；Engine
组装顺序调整为先构造 Worker，再用实际容量创建 `KVCacheManager`。Executor 的
recording-worker 边界不构造模型，无需改变；真实 Engine fixture 通过 initializer 构造
随机 tiny `LlamaModel`。

2026-07-21 Phase 3 分层修正后远端验证：Worker、Executor 和随机模型 Engine
定向测试为 `85 passed in 4.21s`，其中 CUDA 用例实际执行 warmup/profile 和容量推导；
完整默认测试为 `428 passed, 1 skipped in 5.35s`。

## 9. Phase 4：Loaded Model Engine 集成

### 最小端到端

- [x] 从 HF-format tiny checkpoint 调用 `load_model()`。
- [x] 由 ModelRunner initializer 调用 `load_model()`，并接入 SingleProcessExecutor。
- [x] Engine 从 tokenized request 生成至少一个 greedy token。
- [x] output token 通过 request ID 返回。
- [x] EOS、max tokens 和 max model length 行为不回归。

### Chunked prefill 和 decode

- [x] 一次 full prefill 与多步 chunked prefill 产生相同首 token。
- [x] 多 token greedy decode 与 Transformers 逐 token 一致。
- [x] mixed decode/prefill batch 与逐 request 独立执行一致。
- [x] 测试至少包含 GQA 配置。
- [x] 不通过文本“看起来合理”判断正确性。

### KV 生命周期

- [x] prefix cache on/off 的完整 output token IDs 一致。
- [x] prefix hit 确实复用相同 physical block IDs。
- [x] preemption/recompute 与无抢占 output token IDs 一致。
- [x] physical block 释放复用后无 stale KV 污染。
- [x] 请求结束和 abort 后没有 logical block reference 泄漏。

`tests/integration/test_loaded_llama_engine.py` 使用标准 GQA tiny checkpoint，且
`load_model()` 只在 ModelRunner initializer 内调用。CPU 用例逐 token 与 Transformers
full-sequence greedy reference 对拍，并覆盖 full/chunked prefill、mixed batch、prefix
hit、preemption/recompute、physical block reuse、EOS/length/model length 和 abort。
GPU FP16 用例在 Tesla T4 上同时覆盖 checkpoint placement、warmup/profile、KV cache
allocation 和两步 greedy decode。

2026-07-21 Phase 4 远端验证：新增 loaded-engine 测试为 `10 passed in 3.53s`；
Model Loader、Models、Worker、Executor、loaded logits 和 loaded Engine 定向回归为
`170 passed in 5.14s`；完整默认测试为 `438 passed, 1 skipped in 7.82s`。

## 10. Phase 5：公开 Llama Checkpoint 验收

- [x] 通过固定 repo id 和 commit SHA 调用 Hub `load_model()`。
- [x] 首次下载和已有 cache 两种路径都能加载。
- [x] config 与 checkpoint 实际来自同一 commit。
- [x] 模型可移动到远端 GPU 指定 dtype。
- [x] 固定 token IDs 的 prefill logits 与 Transformers 对拍。
- [x] 至少生成 4 个 greedy tokens 并逐 token 一致。
- [x] token budget 强制 chunked prefill 后结果仍一致。
- [x] 记录运行时间、CPU peak memory、GPU peak memory 和 checkpoint 大小。
- [x] 测试结束后不主动删除共享 Hub cache。

首次验收前远端 cache 只有 20KB config；2,200,119,864-byte 权重首次下载约 179 秒，
未在测试后删除。固定 SHA 路径随后使用 `local_files_only=True`，并额外在
`HF_HUB_OFFLINE=1` 下验证完全离线重载。Tesla T4 FP16 实测 cached load 约 9.8 秒，
CPU peak RSS 7,300,128,768 bytes，GPU peak allocated 4,436,969,984 bytes，Runner
warmup peak activation 8,908,288 bytes，可用于 KV cache 的预算为
11,606,777,548 bytes；测试用例以 64 blocks 上限避免验收进程占满共享 GPU。
环境为 Python 3.12.3、PyTorch 2.8.0、Transformers 4.57.6、safetensors 0.8.0、
huggingface-hub 0.36.2，以及 15,360MiB Tesla T4（driver 580.126.09）。

真实 22-layer checkpoint 的 FP16 logits 最大绝对误差实测为 `0.01758`，因此公开模型
使用独立 `atol=rtol=2e-2`；4 个 greedy token 仍逐 token 严格相等。显式模型集合为
`2 passed, 1 deselected in 29.20s`，完全离线 cache 验收为 `1 passed in 24.55s`。
Phase 5 完成后的 Model Loader + Models 定向回归为 `79 passed in 1.49s`；完整默认
回归为 `442 passed, 2 skipped in 7.74s`，两个显式外部模型用例按约定 skip。

## 11. 错误与兼容性边界

- [x] 不支持的真实 Llama config 在模型执行前给出明确错误。
- [x] checkpoint parameter 名称不匹配时错误包含 parameter name。
- [x] tensor shape 不兼容时保留 PyTorch shape 错误和模型来源上下文。
- [x] Hub 网络或下载失败不伪装成模型数值错误。
- [x] ModelRunner device/dtype placement 失败不回退到其他 device/dtype。
- [x] Scheduler `max_model_len` 不得超过模型支持的最大 position；本阶段各验收 fixture
      明确保证不超过对应 checkpoint limit，不在 Scheduler 中引入模型依赖。

本阶段不新增一套完整 missing/unexpected/duplicate 状态机；只有真实 checkpoint 暴露
无法定位的问题时，才以最小改动补充 Loader/Model 的错误上下文。

## 12. 测试文件建议

### 新增

- [x] `tests/integration/test_loaded_llama.py`：本地 HF-format checkpoint 数值对拍。
- [x] `tests/integration/test_loaded_llama_engine.py`：loaded model Engine 语义。
- [x] `tests/integration/test_hub_llama.py`：显式公开模型测试。

### 更新

- [x] `tests/worker/test_worker.py`：initializer 转交、实际 CacheConfig 和生命周期编排。
- [x] `tests/worker/test_model_runner_execution.py`：模型初始化、warmup/profile、KV 分配和执行。
- [x] `tests/integration/test_real_model_engine.py`：保留随机模型快速路径并适配新 Worker。
- [x] `pyproject.toml`：注册外部模型测试 marker。
- [x] 必要时更新 `tests/conftest.py`，集中 tiny HF checkpoint fixture。

### 回归要求

- [x] Model Loader 定向测试通过。
- [x] Models 定向测试通过。
- [x] Worker/Executor 定向测试通过。
- [x] CPU 非网络测试通过。
- [x] GPU 非网络测试通过。
- [x] 显式公开模型测试通过。
- [x] 完整默认测试通过，外部模型测试按约定 skip。

## 13. 建议提交顺序

1. `test: create Hugging Face tiny Llama checkpoints`
2. `test: verify loaded Llama parameters and logits`
3. `refactor: move model lifecycle into model runner`
4. `test: run loaded Llama through engine`
5. `test: verify pinned Hub Llama checkpoint`
6. `docs: complete real Llama integration milestone`

Worker/ModelRunner 构造变更与对应测试应在同一提交内保持完整测试可运行。公开模型测试
不得与模型生命周期重构混在同一提交中。

## 14. 远端验证

测试前从本地仓库根目录同步：

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/
```

建议在恢复测试后按以下顺序执行：

```bash
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/model_loader tests/models'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/worker tests/executor'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/integration/test_loaded_llama.py tests/integration/test_loaded_llama_engine.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m "not model"'
ssh aliyun-ecs \
  'cd /root/micro-vllm && MICRO_VLLM_TEST_MODEL=1 .venv/bin/python -m pytest -q -m model tests/integration/test_hub_llama.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q'
```

- [x] 记录 Python、PyTorch、Transformers、safetensors 和 huggingface-hub 版本。
- [x] 记录公开模型 repo、commit、dtype 和硬件。
- [x] 本地 HF-format checkpoint tests 通过。
- [x] Worker/Engine integration tests 通过。
- [x] CPU/GPU 默认测试通过。
- [x] 公开 checkpoint 显式测试通过。
- [x] 完整默认测试通过。

## 15. 完成标准

- [x] V1.4B Model Loader 现有测试先完成远端验证。
- [x] 标准 HF-format Llama checkpoint 能被完整加载。
- [x] loaded parameters、hidden states 和 logits 与 Transformers 对拍。
- [x] ModelRunner 执行 initializer 加载的 model，不重新随机构造模型。
- [x] loaded model 可通过 Engine 完成 chunked prefill 和多 token decode。
- [x] greedy token IDs 与 Transformers 逐 token 一致。
- [x] loaded model 的 prefix cache、preemption 和 block reuse 语义正确。
- [x] 至少一个固定 commit 的公开 Llama-family checkpoint 通过 GPU smoke test。
- [x] 快速测试不依赖网络，公开模型测试可显式启用。
- [x] `docs/requirements/v1-core-engine.md` 按实际证据更新。

完成后进入 V1.4D：Qwen dense model、模型专属 config/parameter 语义和真实
checkpoint 对拍。
