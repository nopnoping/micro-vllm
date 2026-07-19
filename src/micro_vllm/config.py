"""Top-level configuration objects shared across micro-vLLM components."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Device-independent limits enforced by the scheduler."""

    max_num_batched_tokens: int
    max_num_seqs: int

    def __post_init__(self) -> None:
        for field_name in ("max_num_batched_tokens", "max_num_seqs"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
