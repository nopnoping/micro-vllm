"""Immutable execution-plan contracts produced by the scheduler."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """Immutable snapshot of one request scheduled for a model run."""

    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    requires_sampling: bool
    block_table: tuple[int | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "block_table", tuple(self.block_table))


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """Immutable execution plan produced for one scheduler step."""

    scheduled_requests: tuple[ScheduledRequest, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scheduled_requests",
            tuple(self.scheduled_requests),
        )

    @property
    def total_num_scheduled_tokens(self) -> int:
        """Total number of tokens scheduled across all requests."""

        return sum(
            request.num_scheduled_tokens for request in self.scheduled_requests
        )
