# V1.4A Safetensors Checkpoint Source 实现清单

## 1. 目标

实现本地 Hugging Face safetensors checkpoint 的最小读取链路，为后续
Model Loader、参数映射和真实模型加载提供稳定的 tensor source：

```text
local model directory
    -> checkpoint discovery
    -> single-file or sharded manifest
    -> one shard opened at a time
    -> Iterator[(weight_name, CPU tensor)]
    -> parameter loader（下一阶段）
```

本阶段完成后，调用方应能用同一个接口遍历单文件和多 shard checkpoint，
且 source 不会主动将整个 checkpoint 同时加载到 CPU 内存。

## 2. 范围

### 本阶段实现

- [x] 支持本地目录中的单文件 `model.safetensors`。
- [x] 支持 `model.safetensors.index.json` 描述的多 shard checkpoint。
- [x] 以确定性顺序逐个产出 `(weight_name, tensor)`。
- [x] 多 shard 模式一次只打开一个 shard。
- [x] tensor 在 CPU 上读取，保留 checkpoint 中的原始 dtype 和 shape。
- [x] 对缺失文件、非法 index、缺失或多余 tensor 给出带上下文的错误。
- [x] 为单文件、多 shard、流式生命周期和失败路径提供 CPU 单元测试。

### 本阶段不实现

- Hugging Face Hub 下载、缓存、revision 或鉴权。
- PyTorch `.bin`、GGUF 或其他 checkpoint 格式。
- `config.json` 解析和 architecture 选择。
- checkpoint 名称到模型参数名称的映射。
- 参数复制、dtype/device 转换或模型初始化。
- tied embedding、packed QKV、packed gate/up 的加载语义。
- TP/PP shard、量化权重或远端 tensor source。
- 通用插件式 checkpoint source 框架。

## 3. 核心职责和不变量

### Source 负责

- [x] 解析本地 checkpoint 布局。
- [x] 建立不可变、确定性的 shard/weight 遍历顺序。
- [x] 验证 index 与实际 safetensors 文件一致。
- [x] 管理 safetensors 文件句柄生命周期。
- [x] 逐个向调用方暴露原始 CPU tensor。

### Source 不负责

- [x] 不导入或识别具体模型类。
- [x] 不访问 Worker、ModelRunner、Scheduler 或 KV Cache。
- [x] 不解释参数是否 tied、packed、replicated 或 sharded。
- [x] 不修改 tensor 的 dtype、shape、device 或数值。
- [x] 不保存已经产出的全部 tensor。

### 生命周期不变量

- [x] 创建 source 或 manifest 时不读取 tensor 数据。
- [x] 开始迭代前只做文件和 metadata 级发现。
- [x] 单文件只在实际迭代期间打开。
- [x] 多 shard 任意时刻最多打开一个 shard。
- [x] 一个 shard 的迭代结束后，在打开下一个 shard 前关闭当前文件。
- [x] 调用方中断或关闭 generator 时，当前文件句柄仍被可靠关闭。
- [x] source 只保证自身不缓存全部 tensor；调用方是否保留 tensor 由调用方负责。

## 4. 最小接口

目标模块现为：`src/micro_vllm/model_loader/checkpoint.py`。

首版只暴露一个具体 safetensors source，不预先定义多格式抽象基类。建议接口：

```python
class SafetensorsCheckpoint:
    def __init__(self, model_path: str | Path) -> None: ...

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]: ...
```

如实现中需要保存发现结果，可以增加内部 frozen dataclass，例如 shard path 和该
shard 对应的 weight names；该类型不作为公共扩展协议。

接口约束：

- [x] `model_path` 首版只接受本地模型目录。
- [x] 构造函数规范化路径，但不打开 tensor 文件。
- [x] `iter_tensors()` 每次调用创建独立 iterator，不复用半消费状态。
- [x] 返回名称使用 checkpoint 原始名称，不做模型侧重命名。
- [x] 返回 tensor 位于 CPU，dtype 和 shape 与文件一致。
- [x] 不暴露 safetensors 内部 file handle。
- [x] checkpoint reader 保持为 `model_loader` 内部实现。

## 5. Phase 0：依赖和测试基线

### Runtime dependency

- [x] 在 `pyproject.toml` 中显式增加兼容范围明确的 `safetensors` 依赖。
- [x] 不依赖 Transformers 的传递安装来获得 safetensors。
- [x] 不为本阶段增加 `huggingface_hub`。
- [x] 在远端确认 safetensors 版本和 PyTorch 兼容。

### 测试目录

- [x] 新建 `tests/model_loader/test_checkpoint.py`。
- [x] 测试使用临时目录生成 tiny safetensors，不依赖网络和外部模型。
- [x] 测试 tensor 足够小，不把 checkpoint IO 测试变成性能测试。
- [x] 记录开始实现前的远端完整测试基线。

2026-07-20 远端基线：Python 3.12.3、PyTorch 2.8.0+cu128、safetensors
0.8.0；实现前完整测试为 `345 passed in 6.46s`。Phase 0 完成后定向测试为
`1 passed in 1.79s`，完整测试为 `346 passed in 6.40s`。

## 6. Phase 1：Checkpoint Discovery

### 单文件发现

- [x] 模型目录包含 `model.safetensors` 且没有 index 时识别为单文件布局。
- [x] 模型目录不存在时抛出 `FileNotFoundError`。
- [x] `model_path` 不是目录时给出明确错误。
- [x] 既没有单文件也没有 index 时给出包含模型路径的错误。

### Sharded index 发现

- [x] 识别 `model.safetensors.index.json`。
- [x] index 顶层必须是 JSON object。
- [x] `weight_map` 必须存在且为非空 object。
- [x] weight name 和 shard filename 必须是非空字符串。
- [x] shard filename 必须解析到模型目录内的普通文件。
- [x] index 引用的每个 shard 必须存在。
- [x] index 中重复或冲突的定义被明确拒绝。

### 布局歧义

- [x] 同时存在 `model.safetensors` 和 sharded index 时明确拒绝，不静默选择。
- [x] 未被 index 引用的其他 `.safetensors` 文件不参与遍历。
- [x] discovery 结果不依赖文件系统返回顺序。

2026-07-20 Phase 1 远端验证：定向测试 `19 passed in 1.86s`，完整测试
`364 passed in 6.49s`。

## 7. Phase 2：确定性 Tensor Iteration

### 单文件

- [x] 使用 `safetensors.safe_open(..., framework="pt", device="cpu")`。
- [x] weight names 使用稳定排序。
- [x] 每个 tensor 恰好产出一次。
- [x] 空 safetensors 文件被明确拒绝。
- [x] tensor 名称、dtype、shape 和数值与写入内容一致。

### 多 shard

- [x] shard 按规范化 filename 稳定排序。
- [x] 每个 shard 内的 weight names 稳定排序。
- [x] 只产出 `weight_map` 声明的 tensor。
- [x] 每个声明的 weight 恰好产出一次。
- [x] shard 中缺少 index 声明的 tensor 时明确失败。
- [x] shard 中存在 index 未声明的 tensor 时明确失败。
- [x] 同一实际 tensor 不会因错误 index 被重复产出。

### 文件句柄

- [x] `safe_open` 始终位于明确的 context manager 内。
- [x] 当前 shard 消费完毕后才打开下一个 shard。
- [x] 正常完成、读取异常和 generator 提前关闭均会释放当前句柄。
- [x] 测试能观察到最大并发打开 shard 数为 1。

2026-07-20 Phase 2 远端验证：定向测试 `28 passed in 1.90s`，完整测试
`373 passed in 6.44s`。

## 8. Phase 3：错误语义

首版优先使用带清晰信息的内建异常，不建立大而全的异常继承层次。

- [x] 路径或 shard 缺失使用 `FileNotFoundError`。
- [x] layout、JSON schema 或 index/file 不一致使用 `ValueError`。
- [x] 错误消息包含 model path、index path、shard 或 weight name 中的相关上下文。
- [x] safetensors 文件损坏时保留原始异常作为 cause。
- [x] 一个 iterator 失败后不影响重新创建 iterator 并再次尝试。
- [x] 失败路径不会遗留打开的文件句柄。

2026-07-20 Phase 3 远端验证：定向测试 `32 passed in 1.91s`，完整测试
`377 passed in 6.50s`。

不要求 source 或 model loader 汇总模型参数层面的 missing、unexpected 或 shape
mismatch；模型按名称直接复制参数。这里的 missing/unexpected 只描述 index 与
safetensors 文件内容不一致。

## 9. 测试清单

### Discovery

- [x] 不存在的 model directory。
- [x] model path 是普通文件。
- [x] 空目录。
- [x] 合法单文件布局。
- [x] 合法 sharded index 布局。
- [x] 单文件与 index 同时存在。
- [x] 非法 JSON 或缺少 `weight_map`。
- [x] index 引用不存在的 shard。
- [x] shard 路径试图越过 model directory。

### Single-file iteration

- [x] 多个不同名称 tensor 按确定性顺序产出。
- [x] dtype、shape 和数值保持不变。
- [x] tensor 位于 CPU。
- [x] 每次调用 `iter_tensors()` 都能从头完整遍历。
- [x] 空文件被拒绝。

### Sharded iteration

- [x] 两个以上 shard 按确定性顺序产出。
- [x] index 的 JSON key 顺序不影响结果顺序。
- [x] 每个 weight 只出现一次。
- [x] index 声明但 shard 缺少 weight。
- [x] shard 存在 index 未声明的 weight。
- [x] 任意时刻只打开一个 shard。
- [x] 消费部分 iterator 后主动关闭会释放文件句柄。
- [x] 第二个 shard 读取失败时第一个和第二个句柄均正确关闭。

### 回归

- [x] 现有 CPU 测试全部通过。
- [x] 现有 GPU 测试全部通过。
- [x] 测试不访问网络。
- [x] 测试不依赖固定绝对路径。

## 10. 建议提交顺序

每个提交保持可独立审查和通过相应测试：

1. `build: add safetensors runtime dependency`
2. `feat: discover local safetensors checkpoints`
3. `feat: stream single-file checkpoint tensors`
4. `feat: stream sharded checkpoint tensors`
5. `test: cover checkpoint source failures and lifecycle`
6. `docs: complete checkpoint source milestone`

提交不是强制按文件拆分；紧密耦合且无法独立验证的 discovery 与 iteration 改动可以
合并，但不在同一提交中加入 Model Loader 或 parameter loader。

## 11. 远端验证

所有源码编辑只发生在本地。测试前从仓库根目录同步：

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/
```

建议验证顺序：

```bash
ssh aliyun-ecs \
  'cd /root/micro-vllm && \
   .venv/bin/python -m pytest -q tests/model_loader/test_checkpoint.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m "not gpu"'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m gpu'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q'
```

- [x] 记录远端 Python、PyTorch 和 safetensors 版本。
- [x] checkpoint source 定向测试通过。
- [x] CPU 测试通过。
- [x] GPU 测试通过。
- [x] 完整测试通过。

2026-07-20 最终远端验证：checkpoint source 定向测试 `32 passed in 1.91s`；
CPU 测试 `370 passed, 7 deselected in 5.83s`；GPU 测试
`7 passed, 370 deselected in 5.71s`；完整测试 `377 passed in 6.51s`。

## 12. 完成标准

只有以下条件全部满足，V1.4A 才能标记完成：

- [x] 单文件和 sharded safetensors 使用同一个调用接口。
- [x] 输出顺序确定，weight 不重复、不遗漏。
- [x] 多 shard 任意时刻最多打开一个文件。
- [x] source 不缓存全部 checkpoint tensor。
- [x] 文件布局和 index 不一致时错误明确且无句柄泄漏。
- [x] source 不包含模型、参数映射、设备转换或分布式逻辑。
- [x] 定向、CPU、GPU 和完整远端测试全部通过。
- [x] `docs/requirements/v1-core-engine.md` 中对应完成项按证据更新。

完成后下一阶段进入 V1.4B：Model Loader、HF `config.json` 解析和
Llama 参数加载。
