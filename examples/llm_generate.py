"""Run batched generation with the public Qwen3 checkpoint.

After the first download, set ``HF_HUB_OFFLINE=1`` to use only the local
Hugging Face cache.
"""

import torch

from micro_vllm import (
    CacheConfig,
    LLM,
    LLMConfig,
    ModelConfig,
    SamplingParams,
    SchedulerConfig,
)


MODEL_ID = "Qwen/Qwen3-0.6B"


def main() -> None:
    llm = LLM(
        LLMConfig(
            model_config=ModelConfig(
                model=MODEL_ID,
                device="cuda",
                dtype=torch.float16,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                max_num_batched_tokens=64,
                max_num_seqs=2,
            ),
            cache_config=CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.8,
            ),
        ),
    )
    try:
        outputs = llm.generate(
            [
                "Paged attention is",
                "A small language model can",
            ],
            SamplingParams(
                max_tokens=16,
                temperature=0.7,
                top_p=0.9,
                seed=0,
            ),
        )
    finally:
        llm.shutdown()

    for output in outputs:
        print(f"[{output.request_id}] {output.prompt!r}")
        print(f"text: {output.output_text!r}")
        print(f"prompt_token_ids: {output.prompt_token_ids}")
        print(f"output_token_ids: {output.output_token_ids}")
        print(f"finish_reason: {output.finish_reason.name.lower()}")
        print()


if __name__ == "__main__":
    main()
