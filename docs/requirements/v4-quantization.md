# V4：量化需求清单

## 1. 版本目标

建立独立于模型结构的量化扩展机制，覆盖权重加载、量化 Linear 执行和 KV cache dtype 三个核心环节。项目只实现少量代表性方案，用于理解序列化格式、scale metadata、分布式分片和量化 kernel 的边界。

推荐实现顺序：

1. reference INT8 weight-only；
2. 一种 optimized weight-only 方法；
3. 一种外部 checkpoint 格式，如 AWQ 或 GPTQ；
4. FP8 weight/activation；
5. FP8 KV cache。

## 2. 公共抽象

### P0

- [ ] 定义 `QuantizationConfig`，负责从全局配置为 module 选择 quant method。
- [ ] 定义 `QuantMethod` protocol。
- [ ] QuantMethod 至少表达创建参数、加载参数和执行 forward 三个阶段。
- [ ] 模型类不包含 `if quantization == ...` 的格式分支。
- [ ] Linear/Embedding/MoE 等 layer 通过组合 QuantMethod 改变权重表示和执行。
- [ ] 量化配置能够区分 weight dtype、activation dtype、accumulation dtype 和 KV dtype。
- [ ] 未量化 BF16/FP16 路径继续作为稳定 baseline。

建议接口：

```python
class QuantMethod(Protocol):
    def create_weights(self, module, config, parallel_context) -> None: ...
    def load_weight(self, module, name, tensor, parallel_context) -> None: ...
    def apply(self, module, x: Tensor) -> Tensor: ...
```

Model Loader 与模型之间仍传递 `(checkpoint_name, tensor)` stream。具体模型先解析
parameter name，再把量化参数交给对应 layer/QuantMethod；QuantMethod 不读取路径、
Hub 或 checkpoint source。

## 3. Reference Weight Quantization

### P0

- [ ] 实现 per-channel symmetric INT8 weight quantization。
- [ ] 明确量化 axis、scale shape 和零点语义。
- [ ] 支持从浮点 checkpoint 离线/加载时量化。
- [ ] 提供显式 dequant + PyTorch matmul reference 路径。
- [ ] reference 路径不以性能为目标，只用于验证数据布局和误差。
- [ ] 支持 Column Parallel 和 Row Parallel Linear。
- [ ] 支持 packed QKV 和 packed gate/up。
- [ ] 支持 LM head 或明确记录其不量化策略。

### 测试

- [ ] 单层量化/反量化误差在配置阈值内。
- [ ] 模型量化前后 logits 差异有固定测试阈值。
- [ ] greedy generation 可以完成且质量退化符合预期。
- [ ] TP=1 与 TP>1 的 scale/weight shard 语义一致。

## 4. Optimized Weight-Only Backend

### P0

- [ ] 选择并文档化一种 optimized 方案：W8A16 或 W4A16。
- [ ] 实现或封装一种对应 GPU kernel。
- [ ] 定义 kernel 所需 packed layout、group size 和 alignment 限制。
- [ ] 不满足 kernel 限制时给出清晰错误或回退 reference，不静默产生错误结果。
- [ ] optimized backend 与 reference quant method 数值对拍。
- [ ] benchmark 显示 kernel 时间、端到端吞吐和显存变化。

### P1

- [ ] 在 W4A16 方向选择 AWQ 或 GPTQ 中一种 checkpoint 格式。
- [ ] 支持 checkpoint 中 packed weight、scale 和 zero point 的直接加载。
- [ ] 提供格式适配层，将外部命名/布局转换为内部 QuantMethod 所需格式。

## 5. FP8 Weight/Activation

### P1

- [ ] 明确目标硬件和所需 compute capability。
- [ ] 支持一种 FP8 format，例如 E4M3FN。
- [ ] 定义 static scale 或 dynamic scale，首版只选择一种。
- [ ] accumulation 使用 FP16/BF16/FP32 中明确的一种。
- [ ] 不支持硬件上在初始化阶段直接拒绝或回退。
- [ ] TP collective 前后的 dtype 和 scale 语义明确。
- [ ] 提供 FP8 与 BF16 layer/model 数值误差测试。

## 6. KV Cache Quantization

### P0/P1

- [ ] `KVCacheSpec` 增加明确的 KV dtype/quant mode。
- [ ] KV block size 计算考虑量化后的字节数。
- [ ] Worker 根据 spec 初始化正确 dtype 的 KV tensor。
- [ ] Attention Backend 支持 KV 写入时量化和读取时反量化/直接计算。
- [ ] 定义 K、V scale 的粒度和存储位置。
- [ ] prefix cache hash 与共享语义不依赖浮点 KV 内容，保持 token-based hash。
- [ ] 不同 KV dtype 的 cache group 不可错误共享 physical pool。
- [ ] FP8 KV cache 与 FP16/BF16 reference attention 对拍。
- [ ] 记录 KV 显存节省和 attention latency 变化。

## 7. MoE 与分布式兼容

- [ ] QuantMethod 可应用到 expert weight。
- [ ] ExpertPlacement 后每个 EP rank 只加载本地量化 expert 权重。
- [ ] packed expert scale 与 expert ID 映射正确。
- [ ] TP shard 应在明确阶段作用于 packed quantized tensor。
- [ ] 文档明确“先量化后切分”或“先切分后量化”的等价性限制。
- [ ] PP stage 只加载本 stage 的量化 metadata。
- [ ] DP replica 量化配置保持一致。

## 8. 配置、诊断与可观察性

- [ ] 启动日志打印每类 layer 采用的 QuantMethod。
- [ ] 打印权重 dtype、activation dtype、KV dtype 和 scale 策略。
- [ ] 打印不量化或 fallback 的 layer 列表及原因。
- [ ] 统计浮点和量化权重显存。
- [ ] benchmark 同时给出速度、显存和精度差异。
- [ ] 错误信息包含 module name、global/local shape、pack factor 和 group size。

## 9. 明确不包含

- 所有 AWQ/GPTQ/Marlin/Compressed-Tensors 变体；
- 任意组合的混合位宽自动搜索；
- 训练感知量化；
- 在线 calibration pipeline；
- CPU quantized backend；
- 每种 GPU 架构的最优 kernel；
- 低于 4 bit 的实验性格式；
- 量化模型质量评测平台。

## 10. V4 发布验收

- [ ] reference INT8 weight-only 路径端到端可运行。
- [ ] 至少一种 optimized weight quant backend 有正确性和性能结果。
- [ ] 至少一种外部量化 checkpoint 可直接加载，或明确以内部转换格式交付。
- [ ] FP8 KV cache 或等价低精度 KV cache 通过 attention 对拍。
- [ ] TP、PP、MoE 中至少各验证一个量化组合场景。
- [ ] 量化逻辑未侵入 Scheduler 和 Request 核心状态机。
- [ ] 文档提供显存、吞吐和数值误差三维对比。
