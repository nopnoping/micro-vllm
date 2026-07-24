"""Device-independent checks for the attention stage breakdown spike."""

import torch

from benchmarks.spike_attention_breakdown import _extract_request_inputs
from benchmarks.spike_sdpa import SpikeShape, _make_case
from micro_vllm.attention.operators.reference import (
    gather_kv_cache,
    write_kv_cache,
)


def test_extract_request_inputs_keeps_flattened_request_bounds() -> None:
    case = _make_case(
        SpikeShape(
            name="tiny",
            query_length=2,
            context_length=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            block_size=2,
        ),
        device="cpu",
        dtype=torch.float32,
        seed=0,
    )
    write_kv_cache(
        case.key,
        case.value,
        case.layer_kv_cache,
        case.metadata,
    )
    histories = gather_kv_cache(case.layer_kv_cache, case.metadata)

    inputs = _extract_request_inputs(case, histories, 0)

    assert inputs.query.shape == (2, 4, 8)
    assert inputs.cached_key.shape == (4, 2, 8)
    assert inputs.cached_value.shape == (4, 2, 8)
    assert inputs.query_positions.tolist() == [2, 3]
    assert inputs.query_context_starts.tolist() == [0, 0]
    assert inputs.context_start == 0
    assert (inputs.output_start, inputs.output_end) == (0, 2)
