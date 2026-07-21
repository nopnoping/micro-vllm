"""Per-request sampling parameters."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Immutable generation limits and token-selection policy."""

    max_tokens: int
    eos_token_id: int | None = None
    ignore_eos: bool = False
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if self.eos_token_id is not None:
            if type(self.eos_token_id) is not int or self.eos_token_id < 0:
                raise ValueError("eos_token_id must be non-negative")
        if type(self.ignore_eos) is not bool:
            raise ValueError("ignore_eos must be a boolean")

        temperature = _finite_float("temperature", self.temperature)
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        object.__setattr__(self, "temperature", temperature)

        if self.top_k is not None:
            if type(self.top_k) is not int or self.top_k <= 0:
                raise ValueError("top_k must be a positive integer or None")

        top_p = _finite_float("top_p", self.top_p)
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        object.__setattr__(self, "top_p", top_p)

        if self.seed is not None:
            if type(self.seed) is not int or self.seed < 0:
                raise ValueError("seed must be a non-negative integer or None")

        if self.is_greedy and (self.top_k is not None or self.top_p != 1.0):
            raise ValueError(
                "top_k and top_p require a positive temperature"
            )

    @property
    def is_greedy(self) -> bool:
        """Whether this policy selects the maximum-logit token."""

        return self.temperature == 0.0


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized
