"""Token sampling implementations."""

import secrets

import torch

from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.sampling.params import SamplingParams


class Sampler:
    """Apply one immutable token-selection policy per logits row."""

    __slots__ = ()

    def __call__(
        self,
        logits: torch.Tensor,
        request_ids: tuple[str, ...],
        sampling_params: tuple[SamplingParams, ...],
        sample_indices: tuple[int, ...],
        /,
    ) -> ModelRunnerOutput:
        token_ids = tuple(
            self._sample_row(row, params, sample_index)
            for row, params, sample_index in zip(
                logits,
                sampling_params,
                sample_indices,
            )
        )
        return ModelRunnerOutput(
            tuple(
                SampledToken(request_id, token_id)
                for request_id, token_id in zip(request_ids, token_ids)
            )
        )

    def _sample_row(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        sample_index: int,
    ) -> int:
        if params.is_greedy:
            return int(logits.argmax(dim=-1).item())

        filtered_logits = logits.to(torch.float32) / params.temperature
        filtered_logits = _apply_top_k(filtered_logits, params.top_k)
        filtered_logits = _apply_top_p(filtered_logits, params.top_p)
        probabilities = torch.softmax(filtered_logits, dim=-1)
        generator = torch.Generator(device=logits.device)
        base_seed = (
            params.seed if params.seed is not None else secrets.randbits(63)
        )
        generator.manual_seed((base_seed + sample_index) % (2**63))
        return int(
            torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            ).item()
        )


def _apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    top_values, top_indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, -torch.inf)
    return filtered.scatter(0, top_indices, top_values)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p == 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > top_p
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    filtered = torch.full_like(logits, -torch.inf)
    return filtered.scatter(0, sorted_indices, sorted_logits)
