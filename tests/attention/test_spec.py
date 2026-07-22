"""Tests for immutable attention visibility semantics."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.attention.spec import (
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)


@pytest.mark.parametrize(
    ("spec", "positions", "expected_starts"),
    [
        (FullAttentionSpec(), (0, 3, 8), (0, 0, 0)),
        (SlidingWindowSpec(4), (0, 2, 3, 4, 8), (0, 0, 0, 1, 5)),
        (ChunkedLocalSpec(4), (0, 3, 4, 7, 8), (0, 0, 4, 4, 8)),
    ],
)
def test_context_start_is_an_inclusive_visibility_boundary(
    spec: FullAttentionSpec | SlidingWindowSpec | ChunkedLocalSpec,
    positions: tuple[int, ...],
    expected_starts: tuple[int, ...],
) -> None:
    assert tuple(spec.context_start(position) for position in positions) == (
        expected_starts
    )


def test_sliding_window_includes_the_current_token() -> None:
    spec = SlidingWindowSpec(window_size=1)

    assert spec.context_start(7) == 7


@pytest.mark.parametrize(
    ("spec_class", "field_name", "value"),
    [
        (SlidingWindowSpec, "window_size", 0),
        (SlidingWindowSpec, "window_size", -1),
        (SlidingWindowSpec, "window_size", True),
        (SlidingWindowSpec, "window_size", 1.5),
        (ChunkedLocalSpec, "chunk_size", 0),
        (ChunkedLocalSpec, "chunk_size", -1),
        (ChunkedLocalSpec, "chunk_size", True),
        (ChunkedLocalSpec, "chunk_size", 1.5),
    ],
)
def test_bounded_specs_require_strict_positive_integer_sizes(
    spec_class: type[SlidingWindowSpec] | type[ChunkedLocalSpec],
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be a positive integer"):
        spec_class(value)  # type: ignore[arg-type]


def test_specs_are_frozen_value_objects() -> None:
    spec = SlidingWindowSpec(4)

    with pytest.raises(FrozenInstanceError):
        spec.window_size = 8

    assert spec == SlidingWindowSpec(4)
    assert spec != SlidingWindowSpec(8)
