"""Stable executor contracts consumed by the engine."""

from dataclasses import dataclass
from typing import Protocol

from micro_vllm.core.scheduler_output import SchedulerOutput


@dataclass(frozen=True, slots=True)
class SampledToken:
    """One sampled token explicitly associated with its request."""

    request_id: str
    token_id: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelRunnerOutput:
    """Minimal result returned after executing one scheduler plan."""

    sampled_tokens: tuple[SampledToken, ...] = ()

    def __post_init__(self) -> None:
        sampled_tokens = tuple(self.sampled_tokens)
        seen_request_ids: set[str] = set()
        for sampled_token in sampled_tokens:
            if sampled_token.request_id in seen_request_ids:
                raise ValueError(
                    "sampled_tokens must not contain duplicate request_id values"
                )
            seen_request_ids.add(sampled_token.request_id)
        object.__setattr__(self, "sampled_tokens", sampled_tokens)

    def validate_for(self, plan: SchedulerOutput) -> None:
        """Validate sampled-token request IDs against an execution plan."""

        scheduled_by_id = {
            request.request_id: request for request in plan.scheduled_requests
        }
        for sampled_token in self.sampled_tokens:
            scheduled_request = scheduled_by_id.get(sampled_token.request_id)
            if scheduled_request is None:
                raise ValueError(
                    f"unknown sampled token request_id: {sampled_token.request_id}"
                )
            if not scheduled_request.requires_sampling:
                raise ValueError(
                    f"request {sampled_token.request_id} does not require sampling"
                )


class Executor(Protocol):
    """Execution boundary required by the engine control loop."""

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Execute one immutable scheduler plan."""

        ...
