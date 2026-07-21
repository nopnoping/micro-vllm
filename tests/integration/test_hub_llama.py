"""Selection contract for the pinned public Llama checkpoint."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import resource
import time

import pytest
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import LlamaForCausalLM

from micro_vllm.config import SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.model_loader import load_model
from micro_vllm.models.llama import LlamaModel
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.config import (
    KVCacheInitializationConfig,
    ModelExecutionConfig,
)
from micro_vllm.worker.worker import Worker


MODEL_REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "d9128824c0c80111be21424e68086f52413fb413"
MODEL_CHECKPOINT_FILE = "model.safetensors"
MODEL_CHECKPOINT_DTYPE = "bfloat16"
MODEL_PARAMETER_COUNT = 1_100_048_384
MODEL_DOWNLOAD_SIZE_BYTES = 2_200_119_864
MODEL_PROMPT_TOKEN_IDS = (1, 450, 7_483, 310, 3_444)
MODEL_EXECUTION_DEVICE = "cuda"
MODEL_EXECUTION_DTYPE = "float16"
_GPU_ATOL = 2e-2
_GPU_RTOL = 2e-2


class RecordingSingleProcessExecutor(SingleProcessExecutor):
    """Record public-checkpoint execution plans for chunking assertions."""

    __slots__ = ("plans",)

    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.plans: list[SchedulerOutput] = []

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        output = super().execute(plan)
        self.plans.append(plan)
        return output


def test_selected_hub_llama_spec_is_internally_consistent() -> None:
    """Validate the offline parts of the selected-model contract."""
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)
    assert MODEL_CHECKPOINT_DTYPE == "bfloat16"
    assert MODEL_EXECUTION_DEVICE == "cuda"
    assert MODEL_EXECUTION_DTYPE == "float16"
    assert MODEL_DOWNLOAD_SIZE_BYTES > MODEL_PARAMETER_COUNT * 2
    assert all(
        type(token_id) is int and 0 <= token_id < 32_000
        for token_id in MODEL_PROMPT_TOKEN_IDS
    )


@pytest.mark.model
def test_selected_hub_llama_is_public_and_supported() -> None:
    """Verify external metadata without downloading the checkpoint weights."""
    model_info = HfApi().model_info(
        MODEL_REPO_ID,
        revision=MODEL_REVISION,
        files_metadata=True,
    )

    assert model_info.sha == MODEL_REVISION
    assert not model_info.private
    assert not model_info.gated

    safetensors_files = {
        sibling.rfilename: sibling.size
        for sibling in model_info.siblings
        if sibling.rfilename.endswith(".safetensors")
    }
    assert safetensors_files == {
        MODEL_CHECKPOINT_FILE: MODEL_DOWNLOAD_SIZE_BYTES,
    }

    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename="config.json",
            revision=MODEL_REVISION,
        )
    )
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["torch_dtype"] == MODEL_CHECKPOINT_DTYPE
    assert config.get("rope_scaling") is None
    assert "sliding_window" not in config
    assert config.get("attention_bias", False) is False
    assert "auto_map" not in config
    assert _llama_parameter_count(config) == MODEL_PARAMETER_COUNT
    assert max(MODEL_PROMPT_TOKEN_IDS) < config["vocab_size"]


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pinned_hub_llama_gpu_acceptance() -> None:
    """Download once, reload offline, and match Transformers through Engine."""

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    online_load_started = time.perf_counter()
    online_model = load_model(
        MODEL_REPO_ID,
        ReferenceAttentionBackend(),
        revision=MODEL_REVISION,
    )
    online_load_seconds = time.perf_counter() - online_load_started
    assert isinstance(online_model, LlamaModel)
    assert online_model.config.architectures == ("LlamaForCausalLM",)

    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename="config.json",
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_CHECKPOINT_FILE,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    assert config_path.parent == checkpoint_path.parent
    assert config_path.parent.name == MODEL_REVISION
    assert checkpoint_path.stat().st_size == MODEL_DOWNLOAD_SIZE_BYTES

    del online_model
    gc.collect()

    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        max_model_len=64,
    )
    cached_load_seconds: list[float] = []

    def initialize_cached_model() -> LlamaModel:
        started = time.perf_counter()
        model = load_model(
            MODEL_REPO_ID,
            ReferenceAttentionBackend(),
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        cached_load_seconds.append(time.perf_counter() - started)
        assert isinstance(model, LlamaModel)
        return model

    worker = Worker(
        initialize_cached_model,
        KVCacheInitializationConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            max_num_blocks=64,
        ),
        scheduler_config,
        ModelExecutionConfig(device="cuda", dtype=torch.float16),
    )
    assert len(cached_load_seconds) == 1
    model_runner = worker._model_runner
    assert model_runner is not None
    loaded_model = model_runner._model
    assert isinstance(loaded_model, LlamaModel)
    assert next(loaded_model.parameters()).device.type == "cuda"
    assert next(loaded_model.parameters()).dtype is torch.float16

    reference_model = LlamaForCausalLM.from_pretrained(
        MODEL_REPO_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
    ).eval().to("cuda")
    expected_tokens, reference_logits = _reference_greedy_tokens(
        reference_model,
        MODEL_PROMPT_TOKEN_IDS,
        max_tokens=4,
    )

    actual_logits: list[torch.Tensor] = []
    hook = loaded_model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: actual_logits.append(
            output.detach().to(dtype=torch.float32, device="cpu")
        )
    )
    full_tokens, full_plans = _run_engine(
        worker,
        request_id="full",
        token_budget=8,
        max_tokens=4,
    )
    hook.remove()
    chunked_tokens, chunked_plans = _run_engine(
        worker,
        request_id="chunked",
        token_budget=2,
        max_tokens=4,
    )

    torch.testing.assert_close(
        actual_logits[0][0],
        reference_logits[0],
        atol=_GPU_ATOL,
        rtol=_GPU_RTOL,
    )
    assert full_tokens == expected_tokens
    assert chunked_tokens == expected_tokens
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in full_plans
    ] == [5, 1, 1, 1]
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in chunked_plans
    ] == [2, 2, 1, 1, 1, 1]

    metrics = {
        "repo": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "online_load_seconds": round(online_load_seconds, 3),
        "cached_load_seconds": round(cached_load_seconds[0], 3),
        "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "warmup_peak_activation_bytes": (
            model_runner._peak_activation_memory_bytes
        ),
        "available_kv_cache_memory_bytes": (
            model_runner._available_kv_cache_memory_bytes
        ),
        "allocated_kv_blocks": worker.cache_config.num_blocks,
    }
    print("HUB_LLAMA_ACCEPTANCE " + json.dumps(metrics, sort_keys=True))


@torch.inference_mode()
def _reference_greedy_tokens(
    model: LlamaForCausalLM,
    prompt_token_ids: tuple[int, ...],
    *,
    max_tokens: int,
) -> tuple[tuple[int, ...], list[torch.Tensor]]:
    sequence = list(prompt_token_ids)
    generated: list[int] = []
    logits_by_step: list[torch.Tensor] = []
    for _ in range(max_tokens):
        input_ids = torch.tensor(
            [sequence],
            dtype=torch.int64,
            device="cuda",
        )
        logits = model(input_ids, use_cache=False).logits[0, -1]
        logits_by_step.append(logits.to(dtype=torch.float32, device="cpu"))
        token_id = int(logits.argmax(dim=-1).item())
        generated.append(token_id)
        sequence.append(token_id)
    return tuple(generated), logits_by_step


def _run_engine(
    worker: Worker,
    *,
    request_id: str,
    token_budget: int,
    max_tokens: int,
) -> tuple[tuple[int, ...], list[SchedulerOutput]]:
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=1,
            max_model_len=64,
        ),
        KVCacheManager(worker.cache_config),
    )
    executor = RecordingSingleProcessExecutor(worker)
    engine = Engine(scheduler, executor)
    engine.add_request(
        request_id,
        list(MODEL_PROMPT_TOKEN_IDS),
        SamplingParams(max_tokens=max_tokens),
    )
    for _ in range(64):
        if not engine.has_unfinished_requests():
            break
        engine.step()
    else:
        raise AssertionError("Hub Llama engine did not finish within 64 steps")

    request = scheduler.get_request(request_id)
    assert request is not None
    return tuple(request.output_token_ids), executor.plans


def _llama_parameter_count(config: dict[str, object]) -> int:
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    vocab_size = int(config["vocab_size"])
    head_size = hidden_size // num_attention_heads
    key_value_size = num_key_value_heads * head_size

    embedding_parameters = vocab_size * hidden_size
    attention_parameters = (
        hidden_size * hidden_size
        + 2 * hidden_size * key_value_size
        + hidden_size * hidden_size
    )
    mlp_parameters = 3 * hidden_size * intermediate_size
    layer_norm_parameters = 2 * hidden_size
    output_parameters = (
        0 if config["tie_word_embeddings"] else vocab_size * hidden_size
    )
    return (
        embedding_parameters
        + num_layers
        * (attention_parameters + mlp_parameters + layer_norm_parameters)
        + hidden_size
        + output_parameters
    )
