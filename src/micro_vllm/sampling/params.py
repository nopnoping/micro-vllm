"""Per-request sampling parameters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Minimal generation limits used by the control loop."""

    max_tokens: int
    eos_token_id: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if self.eos_token_id is not None:
            if not isinstance(self.eos_token_id, int) or isinstance(
                self.eos_token_id, bool
            ):
                raise TypeError("eos_token_id must be an integer or None")
            if self.eos_token_id < 0:
                raise ValueError("eos_token_id must be non-negative")

        if not isinstance(self.ignore_eos, bool):
            raise TypeError("ignore_eos must be a boolean")
