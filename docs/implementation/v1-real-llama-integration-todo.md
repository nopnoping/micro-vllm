# V1.4C 真实 Llama Checkpoint 对拍与 Engine 集成技术方案

## 1. 目标

将已经实现的 Model Loader 接入真实执行链路，证明从 Hugging Face checkpoint 到
Engine greedy token 的完整路径正确：

```text
local path or pinned HF repo
    -> model_loader.load_model(...)
    -> loaded LlamaModel
    -> Worker
    -> Physical KV Cache + ModelRunner
    -> SingleProcessExecutor
    -> Engine
    -> greedy token IDs
```

本阶段的重点不是继续扩展 Loader，而是验证真实 checkpoint 的参数、logits、token 和
Paged KV 执行语义，并消除 Worker 当前重新构造随机模型的断点。

## 2. 当前基础与验证状态

当前代码已经具备：

- 本地目录和 Hugging Face Hub 模型解析；
- commit-pinned config/checkpoint 下载；
- 单文件和多 shard safetensors 流式读取；
- 静态 architecture registry；
- `LlamaModel.load_weights()` 参数复制和 tied embedding 映射；
- tiny Llama 与 Transformers 的 forward/logits 数值对拍；
- 随机初始化 Llama 的 Engine、chunked prefill、prefix cache 和 preemption 测试。

2026-07-21 按用户决定暂不运行远端测试，因此 V1.4B 当前为“实现完成、待远端
验证”。本清单不把未运行的测试视为已通过。

## 3. 范围

### 本阶段实现

- [ ] 使用 Transformers 生成标准 HF 格式 tiny Llama checkpoint 并通过
      `load_model()` 加载。
- [ ] 校验加载后的参数、hidden states 和 logits 与 Transformers 一致。
- [ ] Worker 接收已构造且已加载的模型，不再根据 config 重新构造模型。
- [ ] 使用 loaded model 打通 Worker、Executor 和 Engine。
- [ ] 真实 checkpoint 的 full/chunked prefill、decode 和 greedy token 与
      Transformers 对拍。
- [ ] 使用一个公开、固定 commit 的小型 Llama-family checkpoint 完成远端 smoke test。
- [ ] 在 loaded model 上验证 prefix cache、preemption/recompute 和 block reuse。

### 本阶段不实现

- tokenizer 或字符串 prompt。
- `LLM.generate()` 高层离线 API。
- temperature、top-k、top-p 或随机采样。
- Qwen、Mistral 或其他模型。
- packed QKV/gate-up 参数布局。
- meta-device、分片初始化或生产级低内存加载。
- 自动 GPU memory profiling 和 block 数推导。
- Hub 登录流程、模型发布或 checkpoint 转换工具。
- TP、PP、DP、量化或 optimized attention。

## 4. 核心设计决策

### 4.1 Worker 接收模型实例

当前 Worker 接收 `ModelConfig`，再通过 Registry 构造一个随机模型；这会丢失
`load_model()` 已经加载的参数。调整后的最小边界：

```python
model = load_model(model_name_or_path, attention_backend, ...)
worker = Worker(model, cache_config, execution_config)
```

- [ ] Worker 不再调用 models registry。
- [ ] Worker 不识别 Llama/Qwen architecture。
- [ ] Worker 从 model 的只读 config 获取 physical KV shape。
- [ ] Worker 将模型移动到 execution device/dtype，并取得其生命周期所有权。
- [ ] Worker 仍创建和拥有 Physical KV Cache、ModelRunner。
- [ ] shutdown 释放 model 和 KV cache 引用，不修改 checkpoint/cache 文件。
- [ ] 不同时保留“config 构造”和“model 注入”两套 Worker 主路径。

如果静态类型需要公共模型契约，只定义 ModelRunner/Worker 实际消费的窄 protocol：
`config`、model call 和 `compute_logits()`；不建立通用模型基类。

### 4.2 Loader 与执行设备分离

- [ ] `load_model()` 继续返回 CPU 默认 dtype、eval mode 模型。
- [ ] checkpoint source 继续只产生 CPU tensor。
- [ ] Worker 是模型 device/dtype placement 的唯一位置。
- [ ] Model Loader 不创建 KV cache，不读取 SchedulerConfig。
- [ ] 本阶段记录 CPU model 和目标 device model 的峰值内存限制，但不引入 meta loading。

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

2026-07-21 Phase 0 本地实现完成。按本轮约定未同步云端、未运行
Python 或 pytest；marker 默认 skip 和外部元数据校验仍待远端验证。

## 6. Phase 1：HF-format Tiny Checkpoint Fixture

- [ ] 使用 Transformers tiny `LlamaConfig` 构造 reference model。
- [ ] 固定 torch seed 并设置 eval mode。
- [ ] 使用 `save_pretrained(..., safe_serialization=True)` 写入临时目录。
- [ ] 保存的 `config.json` 包含受支持 architecture。
- [ ] checkpoint 使用标准 `model.*` 和 `lm_head.weight` 名称。
- [ ] fixture 默认生成单文件 checkpoint。
- [ ] 增加一个强制小 shard size 的多 shard fixture，覆盖 index 路径。
- [ ] fixture 不使用网络，不写入仓库，不依赖进程外缓存。

## 7. Phase 2：加载参数与数值对拍

### 参数

- [ ] `load_model()` 加载单文件 tiny checkpoint。
- [ ] `load_model()` 加载多 shard tiny checkpoint。
- [ ] micro-vLLM 与 Transformers 的参数名称映射集合一致。
- [ ] 每个 untied parameter 数值一致。
- [ ] tied embedding 保持对象 identity 和数值一致。
- [ ] 加载后 model 为 eval mode、CPU 默认 dtype。
- [ ] 两次加载返回互不共享参数的模型实例。

### Forward 和 logits

- [ ] 使用相同 token IDs 和 positions。
- [ ] 为 micro-vLLM 创建等价 physical KV cache 和 metadata。
- [ ] final hidden states 与 Transformers 对拍。
- [ ] logits 与 Transformers 对拍。
- [ ] CPU FP32 使用明确的 `atol`/`rtol`。
- [ ] GPU FP16 使用单独、较宽且有依据的容差。
- [ ] 失败报告包含 checkpoint、dtype、shape 和最大误差。

## 8. Phase 3：Worker 模型注入

- [ ] Worker 构造参数从 `ModelConfig` 改为已构造 model。
- [ ] Worker 不再导入 `get_model_class`。
- [ ] Worker 从 `model.config` 创建 Physical KV Cache。
- [ ] model 缺少执行所需 contract 时在初始化阶段明确失败。
- [ ] Worker 将 loaded CPU model 移动到配置 device/dtype。
- [ ] ModelRunner 收到的 model 与 Worker 拥有的 model 是同一实例。
- [ ] shutdown 行为保持幂等。
- [ ] 更新现有 Worker、Executor 和 Engine fixtures，显式构造 model。
- [ ] 随机 tiny model 测试继续保留，作为快速执行基线。

## 9. Phase 4：Loaded Model Engine 集成

### 最小端到端

- [ ] 从 HF-format tiny checkpoint 调用 `load_model()`。
- [ ] 将 loaded model 注入 Worker 和 SingleProcessExecutor。
- [ ] Engine 从 tokenized request 生成至少一个 greedy token。
- [ ] output token 通过 request ID 返回。
- [ ] EOS、max tokens 和 max model length 行为不回归。

### Chunked prefill 和 decode

- [ ] 一次 full prefill 与多步 chunked prefill 产生相同首 token。
- [ ] 多 token greedy decode 与 Transformers 逐 token 一致。
- [ ] mixed decode/prefill batch 与逐 request 独立执行一致。
- [ ] 测试至少包含 GQA 配置。
- [ ] 不通过文本“看起来合理”判断正确性。

### KV 生命周期

- [ ] prefix cache on/off 的完整 output token IDs 一致。
- [ ] prefix hit 确实复用相同 physical block IDs。
- [ ] preemption/recompute 与无抢占 output token IDs 一致。
- [ ] physical block 释放复用后无 stale KV 污染。
- [ ] 请求结束和 abort 后没有 logical block reference 泄漏。

## 10. Phase 5：公开 Llama Checkpoint 验收

- [ ] 通过固定 repo id 和 commit SHA 调用 Hub `load_model()`。
- [ ] 首次下载和已有 cache 两种路径都能加载。
- [ ] config 与 checkpoint 实际来自同一 commit。
- [ ] 模型可移动到远端 GPU 指定 dtype。
- [ ] 固定 token IDs 的 prefill logits 与 Transformers 对拍。
- [ ] 至少生成 4 个 greedy tokens 并逐 token 一致。
- [ ] token budget 强制 chunked prefill 后结果仍一致。
- [ ] 记录运行时间、CPU peak memory、GPU peak memory 和 checkpoint 大小。
- [ ] 测试结束后不主动删除共享 Hub cache。

## 11. 错误与兼容性边界

- [ ] 不支持的真实 Llama config 在模型执行前给出明确错误。
- [ ] checkpoint parameter 名称不匹配时错误包含 parameter name。
- [ ] tensor shape 不兼容时保留 PyTorch shape 错误和模型来源上下文。
- [ ] Hub 网络或下载失败不伪装成模型数值错误。
- [ ] Worker device/dtype placement 失败不回退到其他 device/dtype。
- [ ] Scheduler `max_model_len` 不得超过模型支持的最大 position；本阶段由组装代码
      显式校验或测试 fixture 明确保证，不在 Scheduler 中引入模型依赖。

本阶段不新增一套完整 missing/unexpected/duplicate 状态机；只有真实 checkpoint 暴露
无法定位的问题时，才以最小改动补充 Loader/Model 的错误上下文。

## 12. 测试文件建议

### 新增

- [ ] `tests/integration/test_loaded_llama.py`：本地 HF-format checkpoint 数值对拍。
- [ ] `tests/integration/test_loaded_llama_engine.py`：loaded model Engine 语义。
- [ ] `tests/integration/test_hub_llama.py`：显式公开模型测试。

### 更新

- [ ] `tests/worker/test_worker.py`：model injection、placement 和 ownership。
- [ ] `tests/integration/test_real_model_engine.py`：保留随机模型快速路径并适配新 Worker。
- [x] `pyproject.toml`：注册外部模型测试 marker。
- [ ] 必要时更新 `tests/conftest.py`，集中 tiny HF checkpoint fixture。

### 回归要求

- [ ] Model Loader 定向测试通过。
- [ ] Models 定向测试通过。
- [ ] Worker/Executor 定向测试通过。
- [ ] CPU 非网络测试通过。
- [ ] GPU 非网络测试通过。
- [ ] 显式公开模型测试通过。
- [ ] 完整默认测试通过，外部模型测试按约定 skip。

## 13. 建议提交顺序

1. `test: create Hugging Face tiny Llama checkpoints`
2. `test: verify loaded Llama parameters and logits`
3. `refactor: inject loaded models into workers`
4. `test: run loaded Llama through engine`
5. `test: verify pinned Hub Llama checkpoint`
6. `docs: complete real Llama integration milestone`

Worker 构造变更与对应测试应在同一提交内保持完整测试可运行。公开模型测试不得与基础
模型注入重构混在同一提交中。

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

- [ ] 记录 Python、PyTorch、Transformers、safetensors 和 huggingface-hub 版本。
- [ ] 记录公开模型 repo、commit、dtype 和硬件。
- [ ] 本地 HF-format checkpoint tests 通过。
- [ ] Worker/Engine integration tests 通过。
- [ ] CPU/GPU 默认测试通过。
- [ ] 公开 checkpoint 显式测试通过。
- [ ] 完整默认测试通过。

## 15. 完成标准

- [ ] V1.4B Model Loader 现有测试先完成远端验证。
- [ ] 标准 HF-format Llama checkpoint 能被完整加载。
- [ ] loaded parameters、hidden states 和 logits 与 Transformers 对拍。
- [ ] Worker 执行的是同一个 loaded model，不重新随机构造模型。
- [ ] loaded model 可通过 Engine 完成 chunked prefill 和多 token decode。
- [ ] greedy token IDs 与 Transformers 逐 token 一致。
- [ ] loaded model 的 prefix cache、preemption 和 block reuse 语义正确。
- [ ] 至少一个固定 commit 的公开 Llama-family checkpoint 通过 GPU smoke test。
- [ ] 快速测试不依赖网络，公开模型测试可显式启用。
- [ ] `docs/requirements/v1-core-engine.md` 按实际证据更新。

完成后进入 V1.4D：Qwen dense model、模型专属 config/parameter 语义和真实
checkpoint 对拍。
