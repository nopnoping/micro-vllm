"""Per-request sampling parameters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Minimal generation limits used by the control loop."""

    max_tokens: int
    eos_token_id: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")
