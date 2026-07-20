"""Token sampling implementations."""

import torch

from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken


class GreedySampler:
    """Stateless argmax sampling over ModelRunner-selected logits rows."""

    def __call__(
        self,
        logits: torch.Tensor,
        request_ids: tuple[str, ...],
        /,
    ) -> ModelRunnerOutput:
        if logits.shape[0] != len(request_ids):
            raise ValueError("logits rows must equal sampling request IDs")

        token_ids = logits.argmax(dim=-1).tolist()
        return ModelRunnerOutput(
            tuple(
                SampledToken(request_id, token_id)
                for request_id, token_id in zip(request_ids, token_ids)
            )
        )
