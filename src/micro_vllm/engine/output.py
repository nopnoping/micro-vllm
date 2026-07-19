"""User-facing engine output data structures."""

from dataclasses import dataclass

from micro_vllm.core.request import FinishReason


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """Token-level snapshot with cumulative output tokens through this step."""

    request_id: str
    output_token_ids: tuple[int, ...]
    finished: bool
    finish_reason: FinishReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_token_ids", tuple(self.output_token_ids))
