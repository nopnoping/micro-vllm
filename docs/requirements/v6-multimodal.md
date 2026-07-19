# V6：ViT 多模态需求清单

## 1. 版本目标

支持以 ViT 为视觉编码器的 decoder-only 多模态模型，建立多模态输入处理、视觉 encoder 调度、encoder 输出缓存和 embedding merge 的核心抽象。

首个支持形态选择 LLaVA 风格：

```text
image -> image processor -> ViT -> projector -> visual embeddings
                                             -> decoder embedding merge
```

V6 首版不从 cross-attention 架构开始，以免同时引入 encoder KV cache 和更复杂的 attention spec。

## 2. 输入数据模型

### P0

- [ ] 定义结构化 `Prompt`，区分 text/token 内容和 multimodal items。
- [ ] 定义 `MultiModalItem`，至少支持 image。
- [ ] 定义 `PlaceholderRange`，记录视觉 embedding 在 decoder 序列中的 token 范围。
- [ ] request 保存预处理后的 decoder token IDs 和 multimodal metadata。
- [ ] 同一请求支持零张、一张或多张图片。
- [ ] 明确图片顺序与 placeholder 顺序的对应关系。
- [ ] 校验 placeholder token 数与视觉 embedding 数一致。
- [ ] 视觉 token 计入 max model length。
- [ ] text-only 请求继续走原有路径，不创建无用视觉 tensor。

## 3. MultiModalProcessor

### P0

- [ ] 定义 `MultiModalProcessor` protocol。
- [ ] 处理图片 resize、normalize、patch/grid 等模型相关逻辑。
- [ ] 将文本中的图片位置转换为 placeholder token 范围。
- [ ] 输出 pixel values、image sizes/grid metadata 和 decoder token IDs。
- [ ] processor 输出可序列化或稳定 hash，供 encoder cache 使用。
- [ ] 支持 variable resolution，或在首版明确限制为 fixed resolution。
- [ ] processor 配置来自模型 config，不散落在 Engine 中。
- [ ] 输入格式错误时在进入 Scheduler 前报告。

### P1

- [ ] 支持预处理结果的 CPU cache。
- [ ] 支持相同图片跨请求复用预处理结果。

## 4. Vision Encoder 与 Projector

### P0

- [ ] 实现或适配至少一种 ViT encoder。
- [ ] 实现对应 multimodal projector。
- [ ] Vision Encoder 与 decoder model 的权重加载规则分离。
- [ ] PP 场景明确 Vision Encoder 所在 stage/rank，首版推荐首 stage。
- [ ] TP 场景明确 ViT 是否复制或分片，首版允许复制。
- [ ] encoder output dtype 与 decoder embedding dtype 兼容。
- [ ] variable image batch 能正确 padding/packing。
- [ ] 视觉 encoder 可以独立运行 reference test。

## 5. EncoderRunner 与 EncoderCache

### EncoderRunner

- [ ] 将需要计算的图片组成 encoder batch。
- [ ] 输出按 request/image ID 可寻址的 visual embeddings。
- [ ] decoder ModelRunner 不负责图片预处理。
- [ ] encoder 失败时能够关联到原 request ID。

### EncoderCache

- [ ] 定义 image/process-config 的稳定 cache key。
- [ ] 缓存 projector 后、可直接 merge 的视觉 embedding。
- [ ] 维护 cache entry size 和引用/生命周期信息。
- [ ] 配置最大 cache memory 或最大 entry 数。
- [ ] cache miss、hit 和 eviction 可观察。
- [ ] 不同模型、processor config 或 dtype 的 entry 不共享。
- [ ] 请求取消或结束不错误释放仍可复用的共享 entry。

## 6. Scheduler 集成

### P0

- [ ] Scheduler 增加独立 `encoder_compute_budget`。
- [ ] decoder token budget 继续沿用 V1 语义。
- [ ] 请求的视觉 encoder 未完成时，不调度依赖其 embedding 的 decoder token 范围。
- [ ] encoder cache hit 的请求可以跳过视觉计算。
- [ ] 多请求 encoder input 可合批。
- [ ] encoder 与 decoder 的执行顺序通过 SchedulerOutput 明确表达。
- [ ] chunked prefill 可在视觉 placeholder 前后分块。
- [ ] 不能在一个不可拆分的视觉 embedding range 内产生不完整 merge。
- [ ] visual token 数计入 scheduled decoder tokens。
- [ ] request preemption 默认不驱逐共享 encoder cache。

### P1

- [ ] encoder compute 与其他请求的 decoder compute 可流水重叠。
- [ ] encoder budget 可以按视觉 token/patch 数而非简单图片数计算。

## 7. Embedding Merge

### P0

- [ ] decoder embedding layer 能识别 PlaceholderRange。
- [ ] text embedding 与 visual embedding 按原序列位置合并。
- [ ] merge 后 positions 与 decoder token positions 一致。
- [ ] visual range 跨 flattened batch 时 request 边界保持正确。
- [ ] 同一请求多图片的 visual embedding 不串位。
- [ ] chunked prefill 只抽取当前 chunk 需要的 visual embedding slice。
- [ ] prefix cache hash 包含能够区分图片内容和 processor 配置的 multimodal identity。
- [ ] 不允许仅凭相同 placeholder token IDs 共享不同图片的 prefix KV。

## 8. 模型与权重加载

- [ ] Model Registry 支持至少一个 LLaVA 风格模型。
- [ ] config adapter 能找到 text config、vision config 和 projector config。
- [ ] safetensors loader 支持 vision、projector 和 language model 参数前缀映射。
- [ ] 支持独立或组合 checkpoint 的明确一种形式。
- [ ] text-only 模式可以跳过 Vision Encoder 初始化，或记录为何不能跳过。
- [ ] 缺失 vision/projector 权重时给出准确诊断。

## 9. 分布式兼容

- [ ] DP replica 拥有独立 EncoderCache，除非未来显式实现共享缓存。
- [ ] PP 下 visual embeddings 能从 encoder 所在 stage 传递到需要 merge 的 stage。
- [ ] TP ranks 对 merge 后 embedding 保持一致。
- [ ] 多模态 request 的 SchedulerOutput 在所有相关 rank 上一致。
- [ ] 明确 V6 首版不支持的 TP/PP/EP 组合并启动时拒绝。

## 10. 可观察性与指标

- [ ] 记录 image preprocessing latency。
- [ ] 记录 vision encoder/projector latency。
- [ ] 记录 encoder batch size 和视觉 token/patch 数。
- [ ] 记录 EncoderCache hit rate、memory 和 eviction。
- [ ] 记录 visual tokens 对 decoder token budget 的占用。
- [ ] trace 展示 image -> placeholder -> encoder -> merge -> decoder 的关联。
- [ ] benchmark 区分 cold-cache 与 warm-cache TTFT。

## 11. 测试

- [ ] text-only 请求在多模态模型上正确生成。
- [ ] 单图片、多图片和 batch 多请求分别测试。
- [ ] 不同分辨率或明确的非法分辨率分别测试。
- [ ] placeholder 数量/顺序错误能够在执行前检测。
- [ ] merge 输出与模型 reference implementation 对拍。
- [ ] chunked prefill 跨视觉 token 边界结果与完整 prefill 一致。
- [ ] 两张不同图片不能错误命中相同 multimodal prefix cache。
- [ ] 相同图片和配置能够命中 EncoderCache。
- [ ] preemption/resume 不重复或遗漏视觉 embedding。
- [ ] 至少一个真实小型多模态 checkpoint 端到端生成。
- [ ] 支持范围内至少验证一种 TP 或 PP 配置。

## 12. 明确不包含

- cross-attention 多模态架构；
- encoder-decoder language model；
- 音频和视频；
- 动态视频帧采样；
- 多模态 KV connector/disaggregated encoder；
- 所有图片 processor 与模型架构；
- 多模态 speculative decoding；
- 跨 DP replica 的共享 EncoderCache。

## 13. V6 发布验收

- [ ] 至少一个 LLaVA 风格 checkpoint 可以处理 text-only、单图和多图输入。
- [ ] visual encoder、projector 和 embedding merge 均有独立 reference test。
- [ ] visual tokens 与文本 token 共同受 decoder token budget 管理。
- [ ] encoder compute 受独立 budget 管理并可合批。
- [ ] EncoderCache 命中不会改变模型输出。
- [ ] chunked prefill、prefix cache 和 preemption 在多模态请求下保持正确。
- [ ] 提供 cold/warm cache 的 TTFT、吞吐和显存报告。
- [ ] V1 Request/Scheduler/Executor 核心契约只通过扩展字段演进，未被重新设计。
