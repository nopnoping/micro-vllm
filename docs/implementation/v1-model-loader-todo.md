# V1.4B Model Loader 实现清单

## 1. 目标

从用户提供的本地目录、Hugging Face repo id 或可选静态短名称映射构造并
加载模型。当前 alias mapping 为空，因此实际可用入口是本地目录和 repo id：

```text
model name/path
    -> local directory or pinned Hugging Face repo
    -> config.json
    -> generic ModelConfig
    -> models registry
    -> registered model class
    -> local or lazily downloaded checkpoint
    -> checkpoint tensor stream
    -> model.load_weights(...)
```

`config` 提供通用只读 `ModelConfig`；`model_loader` 只知道配置和权重在哪里；
`models` 提供 architecture registry、具体模型结构和参数加载语义。

## 2. 边界

### `model_loader` 负责

- [x] 解析本地目录和 HF repo id，并保留经过测试的静态 alias mapping 机制。
- [x] 固定 Hub revision 对应的 commit。
- [x] 读取 HF `config.json` 并构造通用 `ModelConfig`。
- [x] 调用 models registry 获取模型类，不直接识别具体模型。
- [x] 构造模型并传入 checkpoint tensor iterator。
- [x] 发现和流式读取 safetensors checkpoint。

### `config` 负责

- [x] 通用、只读的 `ModelConfig` 及必要字段读取方法。

### `models` 负责

- [x] architecture 名称到模型类的静态 registry。
- [x] 具体模型直接读取自身所需的 config 字段和默认值。
- [x] 模型结构、forward 和 logits 计算。
- [x] checkpoint name 到 parameter 的映射。
- [x] tied name mapping 和直接 parameter copy 语义。
- [x] 将输入 tensor 复制到目标 parameter。

### 禁止

- [x] `models` 不访问路径、网络、Hub API 或 checkpoint 文件。
- [x] `model_loader` 不 import Llama/Qwen 等具体模型。
- [x] `model_loader` 不解释模型专属 config 字段。
- [x] `model_loader` 不解释 Llama/Qwen 的内部 parameter 布局。
- [x] Scheduler、ModelRunner 和 core 不识别 HF architecture。
- [x] 不提供运行时 Registry class 或动态 `register()`。

## 3. 模块结构

```text
model_loader/
    loader.py        resolution, config, Hub download and load_model
    checkpoint.py    safetensors discovery and iteration

config.py            shared immutable ModelConfig

models/
    registry.py      static architecture to model-class mapping
    llama.py         architecture and parameter loading
    qwen.py          Qwen3 architecture and parameter loading
```

当前只有本地目录和 Hugging Face Hub 两种直接分支，不定义
`ModelSource` 协议或 source class；出现第三种真实来源后再评估
是否抽象。

## 4. Phase 0：迁移与命名

- [x] `weights` package 重命名为 `model_loader`。
- [x] 删除 `weights` compatibility shim。
- [x] checkpoint tests 迁移到 `tests/model_loader/`。
- [x] 模型 config/weight tests 迁移到 `tests/models/`。
- [x] 公共入口只导出 `model_loader.load_model`。
- [x] 公共 `config.py` 提供统一的 `ModelConfig`。
- [x] `AGENTS.md` 更新稳定模块边界。

## 5. Phase 1：通用 HF Config

- [x] 读取固定文件名 `config.json`。
- [x] 拒绝缺失、非文件、非法 UTF-8 和非法 JSON。
- [x] 顶层必须是 object。
- [x] `architectures` 必须是非空字符串 list。
- [x] 按配置顺序选择首个支持的 architecture。
- [x] unsupported 错误列出请求值、支持值和 source label。
- [x] 不使用 `model_type` 猜测 architecture。
- [x] 不执行 `auto_map` 或 remote code。
- [x] 参数校验只覆盖配置文件格式和 architecture 选择。

## 6. Phase 2：通用 Config 与模型初始化

- [x] `ModelConfig` 保留通用 metadata，不定义模型专属字段集合。
- [x] `LlamaModel` 直接从 `ModelConfig` 读取构造所需字段。
- [x] `LlamaModel` 假设输入 config 正常，只保留简单默认值。
- [x] 支持 GQA/MHA、HF 默认值和 tied embedding config。
- [x] 不在 Llama 中重复校验 HF 默认配置。
- [x] model loader 只包装 source 上下文，不解释模型字段。

## 7. Phase 3：模型位置解析

- [x] 显式 `Path` 只按本地目录处理。
- [x] 存在的字符串路径按本地目录处理。
- [x] 其他字符串按 alias 或 HF repo id 处理。
- [x] 本地和 HF 分支直接收敛到同一条模型构造与参数加载流程。
- [x] 不为两个简单分支引入 `ModelSource` 协议或 source class。
- [x] alias mapping 与 models registry 相互独立。
- [x] 当前 alias mapping 为空，不宣称存在可用的内置短名称。
- [x] 空模型名明确失败。

## 8. Phase 4：Hugging Face Hub

- [x] 显式添加 `huggingface-hub` runtime dependency。
- [x] config 下载前将 revision 解析为 commit hash。
- [x] 已固定的 40 位 commit SHA 直接使用，不依赖在线 metadata 查询。
- [x] config 和 checkpoint 使用同一 commit hash。
- [x] 构造模型前只下载 config。
- [x] config 校验并构造模型后才下载 checkpoint snapshot。
- [x] 下载范围限制为 config、index 和标准 safetensors 文件。
- [x] 支持标准 cache directory 和 token 参数。
- [x] 支持 `local_files_only`，固定 commit 的已有 cache 可在无网络时加载。
- [ ] 加入经过确认的内置短名称 alias。

## 9. Phase 5：模型参数加载

- [x] `LlamaModel.load_weights()` 接受 tensor iterable，不接受 path/source。
- [x] 当前分离 Q/K/V 和 gate/up 参数直接映射。
- [x] tied embedding/lm_head 两个名称映射到同一 parameter。
- [x] 由 PyTorch `copy_` 处理 dtype/device copy。
- [x] 名称映射后直接查找 parameter 并复制 tensor。
- [x] 不维护 duplicate、missing 或 unexpected 校验状态。
- [x] 未知名称和不可复制 tensor 使用 Python/PyTorch 原生错误。
- [x] loader 返回 CPU 默认 dtype、eval mode 模型。

## 10. 测试结构

- [x] `tests/model_loader/test_config.py`：通用 JSON 和 architecture。
- [x] `tests/model_loader/test_checkpoint.py`：本地 safetensors 生命周期。
- [x] `tests/model_loader/test_resolution.py`：路径、repo id 和 alias resolution。
- [x] `tests/model_loader/test_huggingface.py`：mock Hub 和 revision pinning。
- [x] `tests/model_loader/test_loader.py`：完整本地编排。
- [x] `tests/test_model_config.py`：通用 config contract。
- [x] `tests/models/test_registry.py`：静态 architecture registry。
- [x] `tests/models/test_llama_initialization.py`：Llama config 解释。
- [x] `tests/models/test_llama_weights.py`：Llama 参数语义。

## 11. 验证状态

2026-07-21：Model Loader、Llama/Qwen3 Models 和两个公开 checkpoint cache 路径已在
`aliyun-ecs` 验证。固定 commit 首次下载后，`local_files_only=True` 和
`HF_HUB_OFFLINE=1` 均可从标准共享 cache 重新加载；Hub 网络异常保持原异常类型。
V1.4D 完成后的完整默认回归为 `461 passed, 4 skipped in 9.48s`。

允许验证后执行：

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/

ssh aliyun-ecs \
  'cd /root/micro-vllm && \
   .venv/bin/python -m pytest -q tests/model_loader tests/models'

ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q'
```
