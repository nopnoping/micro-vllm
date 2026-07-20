# V1.4B HF Config 与 Weight Loader 实现清单

## 1. 目标

建立一个边界清晰的本地 Hugging Face 模型加载链路：

```text
weights
    config.json -> architecture-specific model config
    safetensors -> checkpoint tensors
    model config -> models 中的模型定义
    checkpoint tensors -> model parameters
```

`models` 只定义模型 config、layer 组合和 forward；所有文件读取、
架构选择、模型加载编排和 checkpoint 参数复制都属于 `weights`。

## 2. 依赖方向

允许：

```text
weights -> models -> layers
worker  -> weights
worker  -> model/config protocols
```

禁止：

- `models` 读取 model path、JSON 或 safetensors。
- `models` 导入 `weights`。
- `models` 保存 checkpoint、loader 或 registry 状态。
- `worker`、`executor` 或 `core` 识别具体 HF architecture。
- `weights` 修改 scheduler/request/cache 的逻辑状态。

## 3. 简化后的模块职责

### `models/llama.py`

- [x] 定义 frozen `LlamaModelConfig`。
- [x] 定义 `LlamaModel` 和 Llama-specific assembly。
- [x] 只校验模型自身不变量。
- [x] 不读取 HF 文件或解释 checkpoint。

### `weights/config.py`

- [x] 读取本地 `config.json`。
- [x] 校验 JSON 和 `architectures`。
- [x] 将 HF Llama metadata 转换为 `LlamaModelConfig`。
- [ ] 提供简单的 `load_model_config(model_path)` 入口。

### `weights/source.py`

- [x] 发现并流式读取本地 safetensors checkpoint。
- [x] 不解释具体模型参数语义。

### `weights/loader.py`

- [ ] 根据 architecture 选择当前支持的模型加载分支。
- [ ] 构造 `models` 中定义的模型。
- [ ] 将 `SafetensorsSource` 产出的 tensor 复制到模型参数。
- [ ] 报告 missing、unexpected 和 shape mismatch。

首版只有 `LlamaForCausalLM`，直接使用一个小的内部 dispatch 即可。
不提供运行时 `register()`、`ResolvedModel`、delayed factory、entry point 或
module scanning。

## 4. 本阶段不实现

- Hugging Face Hub 下载、缓存、revision 或 remote code。
- `AutoConfig`、`AutoModel` 或 `trust_remote_code`。
- Qwen、Mistral 或其他架构。
- PyTorch `.bin`、GGUF 或量化 checkpoint。
- TP/PP/DP 参数切分。
- tokenizer、generate API 或 sampling 扩展。
- Worker 从 model path 的集成；等 loader 的参数复制语义完成后
  再接入。

## 5. Phase 0：边界修正与基线

- [x] 删除 `models/registry.py`。
- [x] `models/llama.py` 恢复为纯模型定义。
- [x] HF config 读取和转换迁移到 `weights/config.py`。
- [x] config loading 测试迁移到 `tests/weights/`。
- [x] 不新增 runtime dependency。

原始远端基线：`377 passed in 6.46s`。

## 6. Phase 1：读取 HF `config.json`

### 文件与 JSON

- [x] 输入只接受本地模型目录，读取固定文件名 `config.json`。
- [x] 区分目录不存在、model path 非目录、config 缺失和 config
      非文件。
- [x] 非法 JSON、非法 UTF-8 和 duplicate key 明确失败并包含路径。
- [x] 顶层必须是 JSON object。
- [x] 额外合法 HF metadata 允许存在。
- [x] 读取不修改原始文件，也不要求 checkpoint 文件存在。

### `architectures`

- [x] 必须是非空 list，成员必须是非空字符串。
- [x] 不使用 `model_type` 猜测 architecture。
- [x] 不执行 `auto_map` 或 remote code。

原 Phase 0/1 远端验证：config 定向测试 `28 passed in 1.88s`，完整测试
`405 passed in 6.52s`。

## 7. Phase 2：HF Llama Config 转换

### Shape 字段

- [x] 读取 `vocab_size`、`hidden_size`、`intermediate_size`。
- [x] 读取 `num_hidden_layers`、`num_attention_heads`。
- [x] 读取 `max_position_embeddings`。
- [x] 缺失错误包含字段名和 config path。
- [x] 整数字段拒绝 bool、零、负数、float 和字符串。

### Llama 语义

- [x] 缺失 `num_key_value_heads` 时回退到 `num_attention_heads`。
- [x] 支持 HF 默认 `rms_norm_eps`、`rope_theta` 和 `hidden_act=silu`。
- [x] `tie_word_embeddings` 必须为 bool。
- [x] 显式 `head_dim` 必须与计算结果一致。
- [x] 复用 `LlamaModelConfig` 的 head/GQA/RoPE 不变量。
- [x] 拒绝启用的 attention/MLP bias。
- [x] 拒绝非空 RoPE scaling。
- [x] 拒绝 sliding-window 或其他非 full-attention 语义。

原 Phase 2 远端验证：Llama config 定向测试 `121 passed in 2.03s`，
完整测试 `502 passed in 6.78s`。

## 8. Phase 3：简单的 Config Dispatch

公开入口目标：

```python
def load_model_config(model_path: str | Path) -> LlamaModelConfig: ...
```

- [ ] `load_model_config()` 读取 metadata，不读取 safetensors tensor。
- [ ] 首版只支持 `LlamaForCausalLM`。
- [ ] 按 `architectures` 顺序选择首个支持项。
- [ ] unsupported 错误列出请求值和支持值。
- [ ] 返回 frozen `LlamaModelConfig`。
- [ ] 相同目录重复调用不共享可变解析状态。
- [ ] 不引入 Registry class、`ResolvedModel` 或 factory object。

## 9. Phase 4：模型构造与参数加载

目标入口由 `weights/loader.py` 提供：

```python
def load_model(
    model_path: str | Path,
    attention_backend: AttentionBackend,
) -> nn.Module: ...
```

- [ ] loader 调用 `load_model_config()`。
- [ ] loader 构造 `models` 中的模型定义。
- [ ] loader 使用 `SafetensorsSource` 读取 checkpoint。
- [ ] 参数复制保持 checkpoint dtype/shape 校验语义明确。
- [ ] tied、packed QKV 和 packed gate/up 映射由 loader 负责。
- [ ] missing、unexpected 和 shape mismatch 明确失败。
- [ ] 返回 CPU 默认 dtype 模型；device/dtype placement 由 Worker 负责。
- [ ] 模型默认处于 eval mode。

## 10. 边界约束

- `models` 不导入 `weights`。
- `models` 不包含 path、JSON、safetensors 或 loader 逻辑。
- config loading 测试位于 `tests/weights/`。
- `weights.loader` 可以导入具体 model 实现。
- `core`、Scheduler 和 ModelRunner 不导入具体 loader/model 实现。
- Physical KV Cache 仅依赖所需的 structural config contract。

## 11. 远端验证

2026-07-21 边界迁移后定向测试为 `147 passed in 2.14s`，完整测试为
`500 passed in 6.70s`。

每次测试前同步本地工作树：

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/
```

验证顺序：

```bash
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/weights/test_config.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && \
   .venv/bin/python -m pytest -q tests/weights/test_llama_config.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q'
```
