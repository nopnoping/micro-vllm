"""Tokenizer-backed synchronous offline generation facade."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from micro_vllm.config import SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.request import FinishReason
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.engine.engine import Engine
from micro_vllm.engine.output import RequestOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.model_loader import load_model
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.tokenizer import Tokenizer, apply_tokenizer_eos, load_tokenizer
from micro_vllm.worker.config import (
    KVCacheInitializationConfig,
    ModelExecutionConfig,
)
from micro_vllm.worker.worker import Worker


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """Final immutable text and token snapshot for one prompt."""

    request_id: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    output_text: str
    output_token_ids: tuple[int, ...]
    finish_reason: FinishReason

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
        object.__setattr__(self, "output_token_ids", tuple(self.output_token_ids))


class LLM:
    """Own and drive one synchronous tokenizer-backed inference engine."""

    __slots__ = (
        "_closed",
        "_engine",
        "_executor",
        "_request_counter",
        "_scheduler",
        "_scheduler_config",
        "_tokenizer",
        "_worker",
    )

    def __init__(
        self,
        model: str | Path,
        *,
        tokenizer: Tokenizer | None = None,
        tokenizer_name_or_path: str | Path | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        max_model_len: int = 2048,
        max_num_batched_tokens: int = 512,
        max_num_seqs: int = 8,
        block_size: int = 16,
        gpu_memory_utilization: float = 0.9,
        max_num_blocks: int | None = None,
        enable_prefix_cache: bool = False,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if tokenizer is None:
            tokenizer = load_tokenizer(
                model
                if tokenizer_name_or_path is None
                else tokenizer_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
        )

        def initialize_model() -> torch.nn.Module:
            return load_model(
                model,
                ReferenceAttentionBackend(),
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

        worker = Worker(
            initialize_model,
            KVCacheInitializationConfig(
                block_size=block_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_blocks=max_num_blocks,
                enable_prefix_cache=enable_prefix_cache,
            ),
            scheduler_config,
            ModelExecutionConfig(device=device, dtype=dtype),
        )
        scheduler = Scheduler(
            scheduler_config,
            KVCacheManager(worker.cache_config),
        )
        executor = SingleProcessExecutor(worker)

        self._tokenizer = tokenizer
        self._scheduler_config = scheduler_config
        self._worker = worker
        self._scheduler = scheduler
        self._executor = executor
        self._engine = Engine(scheduler, executor)
        self._request_counter = 0
        self._closed = False

    def generate(
        self,
        prompts: str | Sequence[str],
        sampling_params: SamplingParams | Sequence[SamplingParams],
    ) -> tuple[GenerationOutput, ...]:
        """Generate final outputs synchronously and preserve prompt order."""

        if self._closed:
            raise RuntimeError("LLM is shut down")

        prompt_batch = (prompts,) if isinstance(prompts, str) else tuple(prompts)
        if not prompt_batch:
            raise ValueError("prompts must not be empty")
        for index, prompt in enumerate(prompt_batch):
            if not prompt:
                raise ValueError(f"prompt at index {index} must not be empty")

        if isinstance(sampling_params, SamplingParams):
            params_batch = (sampling_params,) * len(prompt_batch)
        else:
            params_batch = tuple(sampling_params)
            if len(params_batch) != len(prompt_batch):
                raise ValueError(
                    "sampling_params length must match prompts length"
                )

        encoded_prompts: list[tuple[int, ...]] = []
        for index, prompt in enumerate(prompt_batch):
            try:
                prompt_token_ids = tuple(self._tokenizer.encode(prompt))
            except Exception as error:
                raise ValueError(
                    f"failed to tokenize prompt at index {index}"
                ) from error
            if not prompt_token_ids:
                raise ValueError(
                    f"tokenizer produced no tokens for prompt at index {index}"
                )
            if len(prompt_token_ids) >= self._scheduler_config.max_model_len:
                raise ValueError(
                    f"prompt at index {index} has {len(prompt_token_ids)} tokens; "
                    "prompt length must be less than max_model_len"
                )
            encoded_prompts.append(prompt_token_ids)

        effective_params = tuple(
            apply_tokenizer_eos(params, self._tokenizer)
            for params in params_batch
        )
        request_ids = tuple(
            self._next_request_id() for _ in prompt_batch
        )
        for request_id, prompt_token_ids, params in zip(
            request_ids,
            encoded_prompts,
            effective_params,
            strict=True,
        ):
            self._engine.add_request(
                request_id,
                list(prompt_token_ids),
                params,
            )

        final_outputs: dict[str, RequestOutput] = {}
        pending_request_ids = set(request_ids)
        while pending_request_ids:
            for output in self._engine.step():
                if output.request_id not in pending_request_ids or not output.finished:
                    continue
                final_outputs[output.request_id] = output
                pending_request_ids.remove(output.request_id)

        outputs: list[GenerationOutput] = []
        for request_id, prompt, prompt_token_ids in zip(
            request_ids,
            prompt_batch,
            encoded_prompts,
            strict=True,
        ):
            final_output = final_outputs[request_id]
            finish_reason = final_output.finish_reason
            if finish_reason is None:
                raise RuntimeError(
                    f"finished request {request_id} has no finish reason"
                )
            try:
                output_text = self._tokenizer.decode(
                    final_output.output_token_ids
                )
            except Exception as error:
                raise ValueError(
                    f"failed to decode output for prompt at index {len(outputs)}"
                ) from error
            outputs.append(
                GenerationOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    output_text=output_text,
                    output_token_ids=final_output.output_token_ids,
                    finish_reason=finish_reason,
                )
            )
        return tuple(outputs)

    def shutdown(self) -> None:
        """Idempotently release the ModelRunner-owned runtime resources."""

        if self._closed:
            return
        self._closed = True
        self._executor.shutdown()

    def _next_request_id(self) -> str:
        request_id = f"request-{self._request_counter}"
        self._request_counter += 1
        return request_id
