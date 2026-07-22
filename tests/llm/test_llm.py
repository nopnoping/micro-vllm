"""Offline LLM facade tests using a real loaded tiny model."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from micro_vllm import (
    CacheConfig,
    GenerationOutput,
    LLM,
    LLMConfig,
    ModelConfig,
    SamplingParams,
    SchedulerConfig,
)
from micro_vllm.core.request import FinishReason


class FakeTokenizer:
    """Small deterministic tokenizer for high-level orchestration tests."""

    def __init__(
        self,
        encodings: dict[str, tuple[int, ...]],
        *,
        eos_token_id: int | None = None,
        failing_prompt: str | None = None,
    ) -> None:
        self.encodings = encodings
        self.eos_token_id = eos_token_id
        self.failing_prompt = failing_prompt
        self.decoded: list[tuple[int, ...]] = []

    def encode(self, text: str, /) -> tuple[int, ...]:
        if text == self.failing_prompt:
            raise RuntimeError("tokenizer backend failed")
        return self.encodings[text]

    def decode(self, token_ids: tuple[int, ...], /) -> str:
        owned_ids = tuple(token_ids)
        self.decoded.append(owned_ids)
        return "tokens:" + ",".join(str(token_id) for token_id in owned_ids)


def make_llm(
    checkpoint_path: Path,
    tokenizer: FakeTokenizer,
    *,
    max_model_len: int = 16,
) -> LLM:
    return LLM(
        LLMConfig(
            model_config=ModelConfig(
                model=checkpoint_path,
                tokenizer=tokenizer,
                device="cpu",
                dtype=torch.float32,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=max_model_len,
                max_num_batched_tokens=8,
                max_num_seqs=2,
            ),
            cache_config=CacheConfig(
                block_size=2,
                num_blocks=16,
            ),
        ),
    )


def test_generate_preserves_batch_order_and_broadcasts_params(
    tiny_hf_llama_checkpoint: tuple[torch.nn.Module, Path],
) -> None:
    _, checkpoint_path = tiny_hf_llama_checkpoint
    tokenizer = FakeTokenizer(
        {
            "first": (1, 2, 3),
            "second": (4, 5),
        },
        eos_token_id=999,
    )
    llm = make_llm(checkpoint_path, tokenizer)

    outputs = llm.generate(
        ["first", "second"],
        SamplingParams(max_tokens=2),
    )

    assert isinstance(outputs, tuple)
    assert [output.prompt for output in outputs] == ["first", "second"]
    assert [output.prompt_token_ids for output in outputs] == [
        (1, 2, 3),
        (4, 5),
    ]
    assert [len(output.output_token_ids) for output in outputs] == [2, 2]
    assert [output.output_text for output in outputs] == [
        "tokens:" + ",".join(map(str, output.output_token_ids))
        for output in outputs
    ]
    assert [output.finish_reason for output in outputs] == [
        FinishReason.LENGTH,
        FinishReason.LENGTH,
    ]
    assert tokenizer.decoded == [
        output.output_token_ids for output in outputs
    ]
    request = llm._scheduler.get_request(outputs[0].request_id)
    assert request is not None
    assert request.sampling_params.eos_token_id == 999
    assert not hasattr(llm, "_model")
    assert llm._worker._model_runner is not None
    assert llm._worker._model_runner._model is not None
    assert (
        llm._scheduler._kv_cache_manager._layout
        is llm._worker.kv_cache_layout
    )

    with pytest.raises(FrozenInstanceError):
        outputs[0].output_text = "changed"  # type: ignore[misc]
    llm.shutdown()


def test_generate_accepts_per_prompt_params_and_sequential_calls(
    tiny_hf_llama_checkpoint: tuple[torch.nn.Module, Path],
) -> None:
    _, checkpoint_path = tiny_hf_llama_checkpoint
    tokenizer = FakeTokenizer(
        {
            "short": (1,),
            "long": (2,),
            "later": (3,),
        }
    )
    llm = make_llm(checkpoint_path, tokenizer)

    first_batch = llm.generate(
        ["short", "long"],
        [SamplingParams(max_tokens=1), SamplingParams(max_tokens=3)],
    )
    second_batch = llm.generate("later", SamplingParams(max_tokens=1))

    assert [len(output.output_token_ids) for output in first_batch] == [1, 3]
    assert len(second_batch) == 1
    assert second_batch[0].prompt == "later"
    all_request_ids = {
        *(output.request_id for output in first_batch),
        second_batch[0].request_id,
    }
    assert len(all_request_ids) == 3
    llm.shutdown()


def test_generate_validates_whole_batch_before_admission(
    tiny_hf_llama_checkpoint: tuple[torch.nn.Module, Path],
) -> None:
    _, checkpoint_path = tiny_hf_llama_checkpoint
    tokenizer = FakeTokenizer(
        {
            "valid": (1, 2),
            "empty-tokens": (),
            "too-long": tuple(range(8)),
        },
        failing_prompt="broken",
    )
    llm = make_llm(checkpoint_path, tokenizer, max_model_len=8)
    params = SamplingParams(max_tokens=1)

    with pytest.raises(ValueError, match="prompts must not be empty"):
        llm.generate([], params)
    with pytest.raises(ValueError, match="prompt at index 0 must not be empty"):
        llm.generate("", params)
    with pytest.raises(ValueError, match="sampling_params length"):
        llm.generate(["valid"], [params, params])
    with pytest.raises(ValueError, match="no tokens.*index 1"):
        llm.generate(["valid", "empty-tokens"], params)
    with pytest.raises(ValueError, match="failed to tokenize prompt at index 1"):
        llm.generate(["valid", "broken"], params)
    with pytest.raises(ValueError, match="index 1 has 8 tokens"):
        llm.generate(["valid", "too-long"], params)

    assert llm._scheduler._requests == {}
    assert llm.generate("valid", params)[0].prompt == "valid"
    llm.shutdown()


def test_shutdown_is_idempotent_and_releases_model_runner(
    tiny_hf_llama_checkpoint: tuple[torch.nn.Module, Path],
) -> None:
    _, checkpoint_path = tiny_hf_llama_checkpoint
    llm = make_llm(checkpoint_path, FakeTokenizer({"prompt": (1,)}))
    model_runner = llm._worker._model_runner
    assert model_runner is not None

    llm.shutdown()
    llm.shutdown()

    assert llm._worker._model_runner is None
    assert model_runner._model is None
    assert model_runner._kv_cache_buffers is None
    with pytest.raises(RuntimeError, match="LLM is shut down"):
        llm.generate("prompt", SamplingParams(max_tokens=1))


def test_generation_output_owns_token_tuples() -> None:
    prompt_ids = [1, 2]
    output_ids = [3, 4]
    output = GenerationOutput(
        request_id="request-0",
        prompt="prompt",
        prompt_token_ids=prompt_ids,  # type: ignore[arg-type]
        output_text="output",
        output_token_ids=output_ids,  # type: ignore[arg-type]
        finish_reason=FinishReason.LENGTH,
    )
    prompt_ids.append(5)
    output_ids.append(6)

    assert output.prompt_token_ids == (1, 2)
    assert output.output_token_ids == (3, 4)
