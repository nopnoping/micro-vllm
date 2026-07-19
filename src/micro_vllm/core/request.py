"""Request state and lifecycle domain objects."""

from dataclasses import dataclass, field
from enum import Enum, auto

from micro_vllm.sampling.params import SamplingParams


def _validate_token_id(field_name: str, token_id: int) -> None:
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise TypeError(f"{field_name} must be an integer")
    if token_id < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_token_ids(field_name: str, token_ids: list[int]) -> None:
    if not isinstance(token_ids, list):
        raise TypeError(f"{field_name} must be a list of integers")
    for index, token_id in enumerate(token_ids):
        _validate_token_id(f"{field_name}[{index}]", token_id)


class RequestStatus(Enum):
    """Scheduler-owned lifecycle state for a request."""

    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()


class FinishReason(Enum):
    """Reason a request entered the finished state."""

    EOS = auto()
    LENGTH = auto()
    ABORTED = auto()


_ALLOWED_STATUS_TRANSITIONS = {
    RequestStatus.WAITING: frozenset({RequestStatus.RUNNING}),
    RequestStatus.RUNNING: frozenset({RequestStatus.PREEMPTED}),
    RequestStatus.PREEMPTED: frozenset({RequestStatus.RUNNING}),
    RequestStatus.FINISHED: frozenset(),
}

_ALLOWED_FINISH_REASONS = {
    RequestStatus.WAITING: frozenset({FinishReason.ABORTED}),
    RequestStatus.RUNNING: frozenset(FinishReason),
    RequestStatus.PREEMPTED: frozenset({FinishReason.ABORTED}),
    RequestStatus.FINISHED: frozenset(),
}


@dataclass(slots=True)
class Request:
    """Mutable request state owned by the scheduler."""

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: FinishReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string")
        if not self.request_id:
            raise ValueError("request_id must not be empty")

        _validate_token_ids("prompt_token_ids", self.prompt_token_ids)
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        _validate_token_ids("output_token_ids", self.output_token_ids)

        # Own the mutable token lists so caller-side changes cannot bypass request
        # lifecycle operations.
        self.prompt_token_ids = list(self.prompt_token_ids)
        self.output_token_ids = list(self.output_token_ids)

        if not isinstance(self.num_computed_tokens, int) or isinstance(
            self.num_computed_tokens, bool
        ):
            raise TypeError("num_computed_tokens must be an integer")
        if not 0 <= self.num_computed_tokens <= self.num_known_tokens:
            raise ValueError(
                "num_computed_tokens must be between zero and num_known_tokens"
            )

        if not isinstance(self.status, RequestStatus):
            raise TypeError("status must be a RequestStatus")
        if self.finish_reason is not None and not isinstance(
            self.finish_reason, FinishReason
        ):
            raise TypeError("finish_reason must be a FinishReason or None")
        if self.status is RequestStatus.FINISHED:
            if self.finish_reason is None:
                raise ValueError("finished request must have a finish_reason")
        elif self.finish_reason is not None:
            raise ValueError("unfinished request must not have a finish_reason")

    @property
    def num_prompt_tokens(self) -> int:
        """Number of tokens in the prompt."""

        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        """Number of sampled output tokens."""

        return len(self.output_token_ids)

    @property
    def num_known_tokens(self) -> int:
        """Total number of prompt and sampled tokens known to the request."""

        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def num_pending_tokens(self) -> int:
        """Number of known tokens that have not been computed yet."""

        return self.num_known_tokens - self.num_computed_tokens

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        """Return an immutable snapshot of all currently known token IDs."""

        return tuple(self.prompt_token_ids) + tuple(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        """Whether the request has reached its terminal state."""

        return self.status is RequestStatus.FINISHED

    def transition_to(self, new_status: RequestStatus) -> None:
        """Apply a legal non-terminal lifecycle transition."""

        if not isinstance(new_status, RequestStatus):
            raise TypeError("new_status must be a RequestStatus")
        if new_status not in _ALLOWED_STATUS_TRANSITIONS[self.status]:
            raise ValueError(
                f"cannot transition request from {self.status.name} "
                f"to {new_status.name}"
            )
        self.status = new_status

    def append_sampled_token(self, token_id: int) -> None:
        """Append one sampled token after all known tokens were computed."""

        _validate_token_id("token_id", token_id)
        if self.status is not RequestStatus.RUNNING:
            raise ValueError("sampled token can only be appended to a running request")
        if self.num_pending_tokens != 0:
            raise ValueError(
                "sampled token cannot be appended while known tokens are pending"
            )
        self.output_token_ids.append(token_id)

    def advance_computed_tokens(self, num_tokens: int) -> None:
        """Commit computation of a positive number of pending tokens."""

        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise TypeError("num_tokens must be an integer")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be greater than zero")
        if self.status is not RequestStatus.RUNNING:
            raise ValueError("computed tokens can only advance for a running request")
        if num_tokens > self.num_pending_tokens:
            raise ValueError("num_tokens must not exceed num_pending_tokens")
        self.num_computed_tokens += num_tokens

    def finish(self, reason: FinishReason) -> None:
        """Move the request to its terminal state with a legal reason."""

        if not isinstance(reason, FinishReason):
            raise TypeError("reason must be a FinishReason")
        if reason not in _ALLOWED_FINISH_REASONS[self.status]:
            raise ValueError(
                f"cannot finish {self.status.name} request with {reason.name}"
            )
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
