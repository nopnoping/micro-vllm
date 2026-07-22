"""Immutable execution-plan contracts produced by the scheduler."""

from dataclasses import dataclass

from micro_vllm.sampling.params import SamplingParams


@dataclass(frozen=True, slots=True)
class CacheGroupBlockTable:
    """One cache group's immutable logical-to-physical block snapshot."""

    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_ids", tuple(self.block_ids))


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """Immutable request snapshot with tables in cache-topology order."""

    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    requires_sampling: bool
    block_tables: tuple[CacheGroupBlockTable, ...]
    sampling_params: SamplingParams | None = None
    sample_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "block_tables", tuple(self.block_tables))


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
