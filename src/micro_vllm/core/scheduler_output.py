"""Immutable execution-plan contracts produced by the scheduler."""

from dataclasses import dataclass

from micro_vllm.sampling.params import SamplingParams


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """Immutable snapshot of one request scheduled for a model run."""

    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    requires_sampling: bool
    block_table: tuple[int | None, ...]
    sampling_params: SamplingParams | None = None
    sample_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "block_table", tuple(self.block_table))
        if self.requires_sampling:
            if self.sampling_params is None or self.sample_index is None:
                raise ValueError(
                    "sampling requests require params and sample_index"
                )
            if type(self.sample_index) is not int or self.sample_index < 0:
                raise ValueError("sample_index must be a non-negative integer")
        elif self.sampling_params is not None or self.sample_index is not None:
            raise ValueError(
                "non-sampling requests cannot carry sampling metadata"
            )


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """Immutable execution plan produced for one scheduler step."""

    scheduled_requests: tuple[ScheduledRequest, ...]
    preempted_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scheduled_requests",
            tuple(self.scheduled_requests),
        )
        object.__setattr__(
            self,
            "preempted_request_ids",
            tuple(self.preempted_request_ids),
        )

    @property
    def total_num_scheduled_tokens(self) -> int:
        """Total number of tokens scheduled across all requests."""

        return sum(
            request.num_scheduled_tokens for request in self.scheduled_requests
        )
