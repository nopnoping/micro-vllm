"""Tests for the high-level tokenizer adapter boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.tokenizer import (
    Tokenizer,
    TransformersTokenizer,
    apply_tokenizer_eos,
    load_tokenizer,
)


class RecordingBackend:
    eos_token_id = 9

    def __init__(self) -> None:
        self.encoded: list[tuple[str, bool]] = []
        self.decoded: list[tuple[list[int], bool, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encoded.append((text, add_special_tokens))
        return [1, 2, 3]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        self.decoded.append(
            (
                list(token_ids),
                skip_special_tokens,
                clean_up_tokenization_spaces,
            )
        )
        return "decoded"


def test_adapter_exposes_narrow_tuple_contract() -> None:
    backend = RecordingBackend()
    tokenizer = TransformersTokenizer(backend)  # type: ignore[arg-type]
    source_ids = [3, 2, 1]

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(source_ids)
    source_ids.append(0)

    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.eos_token_id == 9
    assert encoded == (1, 2, 3)
    assert decoded == "decoded"
    assert backend.encoded == [("hello", True)]
    assert backend.decoded == [([3, 2, 1], True, False)]


def test_apply_tokenizer_eos_only_fills_unspecified_value() -> None:
    tokenizer = TransformersTokenizer(RecordingBackend())  # type: ignore[arg-type]
    unspecified = SamplingParams(max_tokens=4)
    explicit = SamplingParams(max_tokens=4, eos_token_id=7)

    resolved = apply_tokenizer_eos(unspecified, tokenizer)

    assert resolved is not unspecified
    assert resolved.eos_token_id == 9
    assert apply_tokenizer_eos(explicit, tokenizer) is explicit


def test_load_tokenizer_forwards_hf_source_options(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = RecordingBackend()
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_from_pretrained(source: str, **kwargs: object) -> RecordingBackend:
        calls.append((source, kwargs))
        return backend

    monkeypatch.setattr(
        "micro_vllm.tokenizer.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    loaded = load_tokenizer(
        LLMConfig(
            model_config=ModelConfig(
                model="org/model",
                tokenizer_name_or_path=Path("/tmp/model"),
                revision="commit",
                cache_dir=Path("/tmp/cache"),
                token="token",
                local_files_only=True,
            )
        )
    )

    assert loaded._tokenizer is backend
    assert calls == [
        (
            "/tmp/model",
            {
                "revision": "commit",
                "cache_dir": Path("/tmp/cache"),
                "token": "token",
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]


def test_loads_standard_local_fast_tokenizer(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer"
    backend = BackendTokenizer(
        WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    reference = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    reference.save_pretrained(tokenizer_path)

    loaded = load_tokenizer(
        LLMConfig(
            model_config=ModelConfig(
                model=tokenizer_path,
                local_files_only=True,
            )
        )
    )

    assert loaded.eos_token_id == 1
    assert loaded.encode("hello world") == (2, 3)
    assert loaded.decode((2, 3)) == "hello world"
