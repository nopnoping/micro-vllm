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
        if not self.request_id:
            raise ValueError("request_id must not be empty")

        input_token_ids = tuple(self.input_token_ids)
        for index, token_id in enumerate(input_token_ids):
            if token_id < 0:
                raise ValueError(f"input_token_ids[{index}] must be non-negative")

        if self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be non-negative")

        if self.num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be greater than zero")
        if self.num_scheduled_tokens != len(input_token_ids):
            raise ValueError(
                "num_scheduled_tokens must equal the number of input_token_ids"
            )

        block_table = tuple(self.block_table)
        for index, block_id in enumerate(block_table):
            if block_id is not None and block_id < 0:
                raise ValueError(
                    f"block_table[{index}] must be non-negative or None"
                )

        object.__setattr__(self, "input_token_ids", input_token_ids)
        object.__setattr__(self, "block_table", block_table)


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """Immutable execution plan produced for one scheduler step."""

    scheduled_requests: tuple[ScheduledRequest, ...]

    def __post_init__(self) -> None:
        scheduled_requests = tuple(self.scheduled_requests)
        seen_request_ids: set[str] = set()
        for scheduled_request in scheduled_requests:
            if scheduled_request.request_id in seen_request_ids:
                raise ValueError(
                    "scheduled_requests must not contain duplicate request_id values"
                )
            seen_request_ids.add(scheduled_request.request_id)
        object.__setattr__(self, "scheduled_requests", scheduled_requests)

    @property
    def total_num_scheduled_tokens(self) -> int:
        """Total number of tokens scheduled across all requests."""

        return sum(
            request.num_scheduled_tokens for request in self.scheduled_requests
        )
