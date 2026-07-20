# V1.4B HF Config、Model Registry 与 Llama Factory 技术方案

## 1. 目标

在已经完成的 `SafetensorsSource` 之上，建立从本地 Hugging Face 模型目录到
架构专属模型对象的最小构造链路：

```text
local model directory
    -> config.json
    -> architectures
    -> immutable Model Registry
    -> LlamaModelConfig
    -> delayed LlamaModel factory
```

本阶段只回答三个问题：

1. 如何可靠读取 Hugging Face `config.json`；
2. 如何根据 `architectures` 选择 micro-vLLM 模型实现；
3. 如何把原始配置转换为架构专属 config 并构造随机初始化模型。

真实 checkpoint 参数复制留给 V1.4C。完成本阶段不代表已经能够加载真实模型权重。

## 2. 范围

### 本阶段实现

- [x] 从本地模型目录读取和校验 `config.json`。
- [ ] 根据 Hugging Face `architectures` 解析支持的模型实现。
- [ ] 首个注册项支持 `LlamaForCausalLM`。
- [ ] 将 HF Llama 配置转换为 frozen `LlamaModelConfig`。
- [ ] 返回包含 architecture、model config 和延迟 factory 的不可变解析结果。
- [ ] 通过 factory 构造与当前执行链路兼容的 `LlamaModel`。
- [ ] 去除 Physical KV Cache 类型签名对 `LlamaModelConfig` 的直接依赖。
- [ ] 为配置、Registry、factory 和依赖边界提供 CPU 单元测试。

### 本阶段不实现

- checkpoint tensor 遍历之外的参数加载或复制。
- missing、unexpected、shape mismatch 参数检查。
- tied weight 的 checkpoint 加载语义。
- Qwen、Mistral 或其他模型注册项。
- Hugging Face Hub 下载、缓存、revision 或 remote code。
- `AutoConfig`、`AutoModel` 或 `trust_remote_code`。
- tokenizer 和 `LLM.generate()`。
- temperature、top-k、top-p sampling。
- Worker 从 model path 完成真实权重初始化。
- TP/PP/DP、量化或设备自动选择。
- 用户运行时动态注册第三方模型。

## 3. 核心设计

### 3.1 架构专属配置

继续使用 `LlamaModelConfig` 保存 Llama 的完整架构语义，不建立包含所有未来模型字段的
万能 dataclass。

允许定义一个只读、窄的 structural protocol，表达执行层共同需要的 KV shape：

```python
class ModelConfig(Protocol):
    num_hidden_layers: int
    num_key_value_heads: int
    max_position_embeddings: int

    @property
    def head_dim(self) -> int: ...
```

该 protocol 只用于 Registry、Worker 和 Physical KV Cache 的类型边界，不负责解析
架构差异，也不增加 Llama/Qwen 字段的并集。

- [ ] 具体配置仍由模型模块拥有。
- [ ] 公共 protocol 只包含真实消费者共同使用的属性。
- [ ] 不复制一份独立的 KV shape 数据，避免两个权威来源。
- [ ] `LlamaModelConfig` 通过结构类型自然满足 protocol。

### 3.2 解析结果

建议定义不可变解析结果：

```python
@dataclass(frozen=True, slots=True)
class ResolvedModel:
    architecture: str
    config: ModelConfig
    _factory: Callable[[AttentionBackend], nn.Module]

    def build(self, backend: AttentionBackend) -> nn.Module: ...
```

factory 闭包捕获已经校验的架构专属 config，因此 Registry 不需要用 `Any` 将 config
重新传回异构 model constructor。

- [ ] `ResolvedModel` 不保存 checkpoint tensor。
- [ ] resolve 阶段不构造 `nn.Module`。
- [ ] 每次 `build()` 返回独立模型实例。
- [ ] factory 创建 CPU 默认 dtype 模型；device/dtype placement 仍由 Worker 负责。
- [ ] factory 注入 AttentionBackend，不在 Registry 中选择优化 backend。

### 3.3 Registry

Registry 首版使用模块内不可变 mapping：

```text
LlamaForCausalLM -> _resolve_llama(raw_config)
```

- [ ] 不提供运行时 `register()` API。
- [ ] 不扫描 Python modules 或使用 entry points。
- [ ] Registry 只根据 architecture 选择 resolver。
- [ ] resolver 负责架构专属 config 转换和 factory 创建。
- [ ] 新模型只增加 resolver 和静态注册项，不修改 Scheduler/ModelRunner。

## 4. 模块职责

### `models/registry.py`

- [x] 读取本地 `config.json`。
- [x] 校验顶层 JSON 和 `architectures`。
- [ ] 从静态 Registry 选择首个受支持 architecture。
- [ ] 返回 `ResolvedModel`。
- [ ] 报告 unknown architecture 和当前支持列表。

### `models/llama.py`

- [ ] 将 raw HF config 转换为 `LlamaModelConfig`。
- [ ] 校验 micro-vLLM Llama 实现支持的架构语义。
- [ ] 提供 Llama resolver/factory 所需的最小函数。
- [ ] 继续拥有 Llama layer 和 model assembly。

### `worker/kv_cache.py`

- [ ] 只依赖 KV shape 所需的窄 config protocol。
- [ ] 不再导入具体 `LlamaModelConfig`。
- [ ] physical cache shape 和现有行为不变。

### 明确不改动

- [ ] Scheduler 不读取 `config.json`，不识别 architecture。
- [ ] ModelRunner 不增加 Llama/Qwen 分支。
- [ ] `SafetensorsSource` 不读取模型配置。
- [ ] `weights/loader.py` 本阶段继续保持参数加载职责为空。
- [ ] Worker 继续支持现有 tiny `LlamaModelConfig` 构造路径，真实 model path 集成留到
      V1.4C。

## 5. Phase 0：基线与测试结构

- [x] 记录当前远端完整基线：`377 passed`。
- [x] 新建 `tests/models/test_registry.py`。
- [x] Phase 0/1 不涉及 HF Llama 字段转换，无需扩展
      `tests/models/test_llama_config.py`。
- [x] Phase 0/1 不改变 KV config 类型边界，无需扩展
      `tests/worker/test_kv_cache.py`。
- [x] 测试使用临时目录生成 tiny `config.json`，不访问网络。
- [x] 本阶段不新增 runtime dependency。

2026-07-20 远端基线：实现前完整测试为 `377 passed in 6.46s`。

## 6. Phase 1：读取 HF `config.json`

### 文件发现

- [x] 输入路径只接受本地模型目录。
- [x] 读取固定文件名 `config.json`。
- [x] 模型目录不存在时抛出 `FileNotFoundError`。
- [x] model path 不是目录时给出明确错误。
- [x] `config.json` 缺失时抛出带完整路径的 `FileNotFoundError`。
- [x] `config.json` 不是普通文件时明确拒绝。

### JSON 语义

- [x] 非法 JSON 或非法 UTF-8 转换为带 config path 的 `ValueError`。
- [x] 顶层值必须是 JSON object。
- [x] duplicate JSON key 被明确拒绝。
- [x] 未使用的合法 HF metadata 字段允许存在。
- [x] 解析阶段不修改原始文件。

### `architectures`

- [x] `architectures` 必须是非空 list。
- [x] 每个 architecture 必须是非空字符串。
- [x] 按配置中的顺序选择首个受支持 architecture。
- [x] 没有受支持项时错误列出请求值和支持值。
- [x] 不使用 `model_type` 静默猜测 architecture。
- [x] 不执行 `auto_map` 或 remote code。

2026-07-20 Phase 0/1 远端验证：定向测试 `28 passed in 1.88s`，完整测试
`405 passed in 6.52s`。

## 7. Phase 2：HF Llama Config 转换

### 必需 shape 字段

- [ ] 读取 `vocab_size`。
- [ ] 读取 `hidden_size`。
- [ ] 读取 `intermediate_size`。
- [ ] 读取 `num_hidden_layers`。
- [ ] 读取 `num_attention_heads`。
- [ ] 读取 `max_position_embeddings`。
- [ ] 缺失字段错误包含字段名和 config path。
- [ ] 整数字段严格拒绝 bool、零、负数、float 和字符串。

### Llama 语义字段

- [ ] `num_key_value_heads` 缺失时按 HF Llama 语义回退到
      `num_attention_heads`。
- [ ] 支持 `rms_norm_eps`，缺失时使用已记录的 Llama 默认值。
- [ ] 支持 `rope_theta`，缺失时使用已记录的 Llama 默认值。
- [ ] 支持 `hidden_act`，当前只接受 `silu`。
- [ ] 支持 `tie_word_embeddings`，必须为 bool。
- [ ] 可选显式 `head_dim` 必须与当前实现计算结果一致。
- [ ] 继续复用 `LlamaModelConfig.__post_init__()` 的 head/GQA/RoPE 校验。

### 明确拒绝的未支持语义

- [ ] `attention_bias=true` 时明确拒绝。
- [ ] `mlp_bias=true` 时明确拒绝。
- [ ] 非空 `rope_scaling` 时明确拒绝。
- [ ] 配置要求 sliding-window 或其他非 full attention 语义时明确拒绝。
- [ ] 错误不把不支持的配置静默降级为普通 Llama。

只影响训练而推理时关闭的 metadata 不应被过度拒绝；是否拒绝某字段必须对应当前模型
结构或数值语义的真实差异。

## 8. Phase 3：Registry Resolution

- [ ] 注册 `LlamaForCausalLM`。
- [ ] Registry mapping 不可被调用方原地修改。
- [ ] `resolve_model(model_path)` 只读取 metadata，不读取 safetensors tensor。
- [ ] 返回 architecture 精确等于被选中的 HF architecture。
- [ ] 返回 config 的运行时类型是 `LlamaModelConfig`。
- [ ] 同一目录重复 resolve 产生值等价但相互独立的解析结果。
- [ ] unsupported architecture 给出稳定、可测试的错误消息。
- [ ] architecture list 中前项不支持、后项支持时选择后项。
- [ ] architecture list 中多个项支持时选择第一个。

## 9. Phase 4：Llama Model Factory

- [ ] `ResolvedModel.build(backend)` 构造 `LlamaModel`。
- [ ] model 使用 resolved config，不重复解析 `config.json`。
- [ ] `build()` 要求显式传入 backend，不提供隐式 backend 选择。
- [ ] 两次 build 返回不同 model/parameter 实例。
- [ ] 模型默认处于 eval mode。
- [ ] tied embedding 配置下 `lm_head.weight is embed_tokens.weight`。
- [ ] untied 配置下 embedding 和 lm_head 参数相互独立。
- [ ] model parameter names 和 shape 与直接构造 `LlamaModel` 一致。
- [ ] factory 不访问 checkpoint 文件。
- [ ] factory 不把模型移动到 CUDA 或强制转换 dtype。

## 10. 测试清单

### Config file

- [x] 不存在的 model directory。
- [x] model path 是普通文件。
- [x] 缺失或非普通文件 `config.json`。
- [x] 非法 JSON、非法 UTF-8、非 object 顶层和 duplicate key。
- [x] 空、错误类型或包含空字符串的 `architectures`。

### Registry

- [ ] 单个受支持 Llama architecture。
- [ ] unsupported architecture。
- [ ] 多 architecture 的确定性选择。
- [x] 额外 HF metadata 不影响 metadata 解析。
- [x] metadata 解析不打开或要求 checkpoint 文件。
- [ ] Registry 不可变。

### Llama config

- [ ] 所有必需字段正确映射。
- [ ] 每个必需字段缺失时错误明确。
- [ ] 严格类型校验，特别覆盖 bool-as-int。
- [ ] GQA 和 MHA 的 `num_key_value_heads`。
- [ ] 默认字段与显式字段得到等价 config。
- [ ] tied/untied embedding。
- [ ] 不支持 bias、RoPE scaling 和 attention 语义。
- [ ] 现有 `LlamaModelConfig` invariant tests 保持通过。

### Factory 和边界

- [ ] build 返回可执行的 `LlamaModel`。
- [ ] 两次 build 不共享 untied 参数。
- [ ] AttentionBackend 被正确注入。
- [ ] Physical KV Cache 接受满足 protocol 的非 Llama 测试 config。
- [ ] `core`、Scheduler 和 ModelRunner 不导入具体 registry/model 实现。
- [ ] 现有 Worker、Engine 和真实模型执行测试不回归。

## 11. 建议提交顺序

1. `feat: parse Hugging Face Llama config`
2. `feat: resolve models through immutable registry`
3. `feat: build Llama models from resolved config`
4. `refactor: decouple physical KV cache from Llama config`
5. `test: cover model registry and factory boundaries`
6. `docs: complete model registry milestone`

紧密耦合的 config parser 与首个 Registry entry 可以合并，但本阶段提交不得加入真实
checkpoint parameter copy 或 Qwen 实现。

## 12. 远端验证

测试前从本地仓库根目录同步：

```bash
rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  ./ aliyun-ecs:/root/micro-vllm/
```

建议验证顺序：

```bash
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/models/test_registry.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/models/test_llama_config.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q tests/worker/test_kv_cache.py'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m "not gpu"'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q -m gpu'
ssh aliyun-ecs \
  'cd /root/micro-vllm && .venv/bin/python -m pytest -q'
```

- [ ] Registry 定向测试通过。
- [ ] Llama config 定向测试通过。
- [ ] Physical KV Cache 定向测试通过。
- [ ] CPU 测试通过。
- [ ] GPU 测试通过。
- [ ] 完整测试通过。

## 13. 完成标准

- [ ] 合法本地 Llama `config.json` 可解析为 frozen `LlamaModelConfig`。
- [ ] `architectures` 是模型实现选择的唯一权威字段。
- [ ] unsupported 或不兼容配置在模型构造前明确失败。
- [ ] Registry 不可变且不包含 Scheduler/Worker/权重加载逻辑。
- [ ] `ResolvedModel` 延迟构造可执行 Llama model。
- [ ] resolve/build 均不读取 checkpoint tensor。
- [ ] Physical KV Cache 不依赖具体 Llama config 类型。
- [ ] 不增加万能 model config 或运行时插件机制。
- [ ] 定向、CPU、GPU 和完整远端测试全部通过。
- [ ] `docs/requirements/v1-core-engine.md` 按实际完成证据更新。

完成后进入 V1.4C：Parameter Loader、tied weight、参数错误检查和真实 Llama
checkpoint 端到端加载。
