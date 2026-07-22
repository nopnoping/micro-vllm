"""Explicit public-checkpoint tokenizer cache acceptance."""

import json
import time

import pytest

from micro_vllm.config import LLMConfig
from micro_vllm.tokenizer import load_tokenizer


MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROMPT = "Hello, paged attention!"
EXPECTED_EOS_TOKEN_ID = 151_645


@pytest.mark.model
def test_pinned_qwen3_tokenizer_online_and_offline_cache() -> None:
    online_started = time.perf_counter()
    online = load_tokenizer(
        LLMConfig(model=MODEL_REPO_ID, revision=MODEL_REVISION)
    )
    online_seconds = time.perf_counter() - online_started
    token_ids = online.encode(PROMPT)

    cached_started = time.perf_counter()
    cached = load_tokenizer(
        LLMConfig(
            model=MODEL_REPO_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    cached_seconds = time.perf_counter() - cached_started

    assert online.eos_token_id == EXPECTED_EOS_TOKEN_ID
    assert cached.eos_token_id == EXPECTED_EOS_TOKEN_ID
    assert token_ids
    assert cached.encode(PROMPT) == token_ids
    assert online.decode(token_ids) == PROMPT
    assert cached.decode(token_ids) == PROMPT
    print(
        "HUB_QWEN3_TOKENIZER_ACCEPTANCE "
        + json.dumps(
            {
                "repo": MODEL_REPO_ID,
                "revision": MODEL_REVISION,
                "token_ids": token_ids,
                "online_load_seconds": round(online_seconds, 3),
                "cached_load_seconds": round(cached_seconds, 3),
            },
            sort_keys=True,
        )
    )
