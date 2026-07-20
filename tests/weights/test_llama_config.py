"""Tests for loading Hugging Face metadata into a Llama model config."""

from pathlib import Path

import pytest

from micro_vllm.models.llama import LlamaModelConfig
from micro_vllm.weights.config import _llama_config_from_hf


def make_hf_config(**overrides: object) -> dict[str, object]:
    values = {
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 128,
    }
    values.update(overrides)
    return values


def test_hf_llama_config_maps_all_supported_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    config = _llama_config_from_hf(
        make_hf_config(
            vocab_size=320,
            hidden_size=96,
            intermediate_size=192,
            num_hidden_layers=3,
            num_attention_heads=6,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            rope_theta=500_000,
            hidden_act="silu",
            tie_word_embeddings=True,
            head_dim=16,
            transformers_version="4.45.0",
        ),
        config_path,
    )

    assert config == LlamaModelConfig(
        vocab_size=320,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        hidden_act="silu",
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "max_position_embeddings",
    ],
)
def test_hf_llama_config_reports_missing_required_field_and_path(
    tmp_path: Path,
    field_name: str,
) -> None:
    config_path = tmp_path / "config.json"
    raw_config = make_hf_config()
    del raw_config[field_name]

    with pytest.raises(ValueError) as error_info:
        _llama_config_from_hf(raw_config, config_path)

    message = str(error_info.value)
    assert field_name in message
    assert str(config_path) in message


@pytest.mark.parametrize(
    "field_name",
    [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    ],
)
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1"])
def test_hf_llama_config_requires_strict_positive_integer_fields(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be a positive integer: {config_path}",
    ):
        _llama_config_from_hf(
            make_hf_config(**{field_name: value}),
            config_path,
        )


def test_hf_llama_config_defaults_kv_heads_to_attention_heads(
    tmp_path: Path,
) -> None:
    config = _llama_config_from_hf(
        make_hf_config(),
        tmp_path / "config.json",
    )

    assert config.num_key_value_heads == config.num_attention_heads == 4


def test_hf_llama_config_preserves_explicit_gqa_heads(tmp_path: Path) -> None:
    config = _llama_config_from_hf(
        make_hf_config(num_key_value_heads=2),
        tmp_path / "config.json",
    )

    assert config.num_key_value_heads == 2
    assert config.num_query_heads_per_kv_head == 2


def test_hf_llama_defaults_equal_explicit_hf_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    implicit = _llama_config_from_hf(make_hf_config(), config_path)
    explicit = _llama_config_from_hf(
        make_hf_config(
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            hidden_act="silu",
            tie_word_embeddings=False,
            attention_bias=False,
            mlp_bias=False,
            rope_scaling=None,
            head_dim=16,
        ),
        config_path,
    )

    assert implicit == explicit


@pytest.mark.parametrize("field_name", ["rms_norm_eps", "rope_theta"])
@pytest.mark.parametrize(
    "value",
    [True, 0, -1.0, "1e-6", float("nan"), float("inf"), 10**1000],
)
def test_hf_llama_config_requires_positive_finite_numbers(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be a positive number: {config_path}",
    ):
        _llama_config_from_hf(
            make_hf_config(**{field_name: value}),
            config_path,
        )


@pytest.mark.parametrize("hidden_act", ["gelu", "relu", None, 1])
def test_hf_llama_config_rejects_unsupported_hidden_act(
    tmp_path: Path,
    hidden_act: object,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError, match=rf"hidden_act.*silu.*{config_path}"):
        _llama_config_from_hf(
            make_hf_config(hidden_act=hidden_act),
            config_path,
        )


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_hf_llama_config_preserves_tied_embedding_setting(
    tmp_path: Path,
    tie_word_embeddings: bool,
) -> None:
    config = _llama_config_from_hf(
        make_hf_config(tie_word_embeddings=tie_word_embeddings),
        tmp_path / "config.json",
    )

    assert config.tie_word_embeddings is tie_word_embeddings


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_hf_llama_config_requires_boolean_tied_embedding_setting(
    tmp_path: Path,
    value: object,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(
        ValueError,
        match=rf"tie_word_embeddings must be a boolean: {config_path}",
    ):
        _llama_config_from_hf(
            make_hf_config(tie_word_embeddings=value),
            config_path,
        )


@pytest.mark.parametrize("head_dim", [16, None])
def test_hf_llama_config_accepts_matching_or_implicit_head_dim(
    tmp_path: Path,
    head_dim: object,
) -> None:
    config = _llama_config_from_hf(
        make_hf_config(head_dim=head_dim),
        tmp_path / "config.json",
    )

    assert config.head_dim == 16


def test_hf_llama_config_rejects_mismatched_head_dim(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError, match=rf"head_dim must equal.*{config_path}"):
        _llama_config_from_hf(
            make_hf_config(head_dim=32),
            config_path,
        )


@pytest.mark.parametrize("head_dim", [True, 0, -1, 16.0, "16"])
def test_hf_llama_config_requires_strict_positive_integer_head_dim(
    tmp_path: Path,
    head_dim: object,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(
        ValueError,
        match=rf"head_dim must be a positive integer: {config_path}",
    ):
        _llama_config_from_hf(
            make_hf_config(head_dim=head_dim),
            config_path,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"hidden_size": 66},
            "hidden_size must be divisible by num_attention_heads",
        ),
        (
            {"num_key_value_heads": 3},
            "num_attention_heads must be divisible by num_key_value_heads",
        ),
        ({"hidden_size": 60}, "attention head_dim must be even"),
    ],
)
def test_hf_llama_config_reuses_model_config_invariants(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError) as error_info:
        _llama_config_from_hf(make_hf_config(**overrides), config_path)

    assert message in str(error_info.value)
    assert str(config_path) in str(error_info.value)


@pytest.mark.parametrize("field_name", ["attention_bias", "mlp_bias"])
def test_hf_llama_config_rejects_enabled_bias(
    tmp_path: Path,
    field_name: str,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError, match=rf"{field_name}=true.*{config_path}"):
        _llama_config_from_hf(
            make_hf_config(**{field_name: True}),
            config_path,
        )


@pytest.mark.parametrize("field_name", ["attention_bias", "mlp_bias"])
@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_hf_llama_config_requires_boolean_bias_settings(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be a boolean: {config_path}",
    ):
        _llama_config_from_hf(
            make_hf_config(**{field_name: value}),
            config_path,
        )


def test_hf_llama_config_accepts_empty_rope_scaling(tmp_path: Path) -> None:
    config = _llama_config_from_hf(
        make_hf_config(rope_scaling={}),
        tmp_path / "config.json",
    )

    assert config.rope_theta == 10_000.0


def test_hf_llama_config_rejects_non_empty_rope_scaling(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError, match=rf"non-empty rope_scaling.*{config_path}"):
        _llama_config_from_hf(
            make_hf_config(rope_scaling={"rope_type": "linear", "factor": 2.0}),
            config_path,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sliding_window": 128}, "sliding_window is unsupported"),
        ({"use_sliding_window": True}, "use_sliding_window=true is unsupported"),
        ({"attention_type": "sliding_attention"}, "attention_type must be"),
        (
            {"layer_types": ["full_attention", "sliding_attention"]},
            "layer_types must use only 'full_attention'",
        ),
    ],
)
def test_hf_llama_config_rejects_non_full_attention_semantics(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError) as error_info:
        _llama_config_from_hf(make_hf_config(**overrides), config_path)

    assert message in str(error_info.value)
    assert str(config_path) in str(error_info.value)


def test_hf_llama_config_accepts_explicit_full_attention_metadata(
    tmp_path: Path,
) -> None:
    config = _llama_config_from_hf(
        make_hf_config(
            use_sliding_window=False,
            attention_type="full_attention",
            layer_types=["full_attention", "full_attention"],
        ),
        tmp_path / "config.json",
    )

    assert config.num_hidden_layers == 2
