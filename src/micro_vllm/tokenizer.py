"""Tokenizer boundary owned by the high-level offline API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from micro_vllm.sampling.params import SamplingParams


@runtime_checkable
class Tokenizer(Protocol):
    """Narrow tokenizer operations consumed by the offline LLM facade."""

    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str, /) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int], /) -> str: ...


class TransformersTokenizer:
    """Tuple-based adapter around one Hugging Face tokenizer."""

    __slots__ = ("_tokenizer",)

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @property
    def eos_token_id(self) -> int | None:
        """Return the backend EOS token ID, if one is configured."""

        return self._tokenizer.eos_token_id

    def encode(self, text: str, /) -> tuple[int, ...]:
        """Encode raw text with the backend's standard special-token policy."""

        return tuple(
            self._tokenizer.encode(text, add_special_tokens=True)
        )

    def decode(self, token_ids: Sequence[int], /) -> str:
        """Decode output token IDs while omitting model special tokens."""

        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def load_tokenizer(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> TransformersTokenizer:
    """Load a standard HF tokenizer without executing remote code."""

    source = (
        str(model_name_or_path)
        if isinstance(model_name_or_path, Path)
        else model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    return TransformersTokenizer(tokenizer)


def apply_tokenizer_eos(
    sampling_params: SamplingParams,
    tokenizer: Tokenizer,
) -> SamplingParams:
    """Fill an unspecified request EOS from the high-level tokenizer."""

    if sampling_params.eos_token_id is not None:
        return sampling_params
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return sampling_params
    return replace(sampling_params, eos_token_id=eos_token_id)
