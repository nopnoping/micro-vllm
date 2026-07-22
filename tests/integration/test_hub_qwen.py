"""Selection contract for the pinned public Qwen checkpoint."""

from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path
import resource
import time

import pytest
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import Qwen3ForCausalLM

from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig, SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.model_loader import load_model
from micro_vllm.models.qwen import Qwen3Model
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker


MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_CHECKPOINT_FILE = "model.safetensors"
MODEL_CHECKPOINT_DTYPE = "bfloat16"
MODEL_UNIQUE_PARAMETER_COUNT = 596_049_920
MODEL_CHECKPOINT_ELEMENT_COUNT = 751_632_384
MODEL_DOWNLOAD_SIZE_BYTES = 1_503_300_328
MODEL_VOCAB_SIZE = 151_936
MODEL_PROMPT_TOKEN_IDS = (1, 450, 7_483, 310, 3_444)
MODEL_EXECUTION_DEVICE = "cuda"
MODEL_EXECUTION_DTYPE = "float16"
_GPU_ATOL = 5e-2
_GPU_RTOL = 5e-2


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


def test_selected_hub_qwen_spec_is_internally_consistent() -> None:
    """Validate the offline parts of the selected-model contract."""
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)
    assert MODEL_CHECKPOINT_DTYPE == "bfloat16"
    assert MODEL_EXECUTION_DEVICE == "cuda"
    assert MODEL_EXECUTION_DTYPE == "float16"
    assert MODEL_DOWNLOAD_SIZE_BYTES > MODEL_CHECKPOINT_ELEMENT_COUNT * 2
    assert all(
        type(token_id) is int and 0 <= token_id < MODEL_VOCAB_SIZE
        for token_id in MODEL_PROMPT_TOKEN_IDS
    )


@pytest.mark.model
def test_selected_hub_qwen_is_public_and_supported() -> None:
    """Verify external metadata without downloading checkpoint weights."""
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
    assert model_info.safetensors is not None
    assert model_info.safetensors.parameters == {
        "BF16": MODEL_CHECKPOINT_ELEMENT_COUNT,
    }
    assert model_info.safetensors.total == MODEL_CHECKPOINT_ELEMENT_COUNT

    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename="config.json",
            revision=MODEL_REVISION,
        )
    )
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["model_type"] == "qwen3"
    assert config["torch_dtype"] == MODEL_CHECKPOINT_DTYPE
    assert config["vocab_size"] == MODEL_VOCAB_SIZE
    assert config["tie_word_embeddings"] is True
    assert config["use_sliding_window"] is False
    assert config["sliding_window"] is None
    assert config["rope_scaling"] is None
    assert config["attention_bias"] is False
    assert config["head_dim"] == 128
    assert "auto_map" not in config
    assert _qwen3_unique_parameter_count(config) == MODEL_UNIQUE_PARAMETER_COUNT
    assert (
        MODEL_UNIQUE_PARAMETER_COUNT + MODEL_VOCAB_SIZE * config["hidden_size"]
        == MODEL_CHECKPOINT_ELEMENT_COUNT
    )
    assert max(MODEL_PROMPT_TOKEN_IDS) < config["vocab_size"]


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pinned_hub_qwen3_gpu_acceptance() -> None:
    """Download once, reload offline, and match Transformers through Engine."""

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    online_load_started = time.perf_counter()
    online_model = load_model(
        LLMConfig(
            model_config=ModelConfig(
                model=MODEL_REPO_ID,
                revision=MODEL_REVISION,
            )
        )
    )
    online_load_seconds = time.perf_counter() - online_load_started
    assert isinstance(online_model, Qwen3Model)
    assert online_model.config.architectures == ("Qwen3ForCausalLM",)

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

    config = LLMConfig(
        model_config=ModelConfig(
            model=MODEL_REPO_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            device="cuda",
            dtype=torch.float16,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8,
            max_num_seqs=1,
            max_model_len=64,
        ),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            num_blocks=64,
        ),
    )
    cached_load_seconds: list[float] = []

    def initialize_cached_model() -> Qwen3Model:
        started = time.perf_counter()
        model = load_model(config)
        cached_load_seconds.append(time.perf_counter() - started)
        assert isinstance(model, Qwen3Model)
        return model

    worker = Worker(initialize_cached_model, config)
    assert len(cached_load_seconds) == 1
    model_runner = worker._model_runner
    assert model_runner is not None
    loaded_model = model_runner._model
    assert isinstance(loaded_model, Qwen3Model)
    assert next(loaded_model.parameters()).device.type == "cuda"
    assert next(loaded_model.parameters()).dtype is torch.float16
    kv_cache_buffers = model_runner._kv_cache_buffers
    assert kv_cache_buffers is not None
    assert kv_cache_buffers[0].shape[-1] == 128

    reference_model = Qwen3ForCausalLM.from_pretrained(
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

    logits_max_absolute_error = torch.max(
        torch.abs(actual_logits[0][0] - reference_logits[0])
    ).item()
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
        "allocated_kv_blocks": worker.config.cache_config.num_blocks,
        "sampling_logits_max_absolute_error": logits_max_absolute_error,
    }
    print("HUB_QWEN3_ACCEPTANCE " + json.dumps(metrics, sort_keys=True))


@torch.inference_mode()
def _reference_greedy_tokens(
    model: Qwen3ForCausalLM,
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
    config = replace(
        worker.config,
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=1,
            max_model_len=64,
        ),
    )
    scheduler = Scheduler(
        config,
        KVCacheManager(config),
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
        raise AssertionError("Hub Qwen3 engine did not finish within 64 steps")

    request = scheduler.get_request(request_id)
    assert request is not None
    return tuple(request.output_token_ids), executor.plans


def _qwen3_unique_parameter_count(config: dict[str, object]) -> int:
    """Calculate unique dense Qwen3 model parameters with tied embeddings."""
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_hidden_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    query_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim

    embedding_parameters = vocab_size * hidden_size
    attention_parameters = (
        hidden_size * query_size
        + 2 * hidden_size * kv_size
        + query_size * hidden_size
    )
    query_key_norm_parameters = 2 * head_dim
    mlp_parameters = 3 * hidden_size * intermediate_size
    layer_norm_parameters = 2 * hidden_size
    final_norm_parameters = hidden_size
    output_parameters = (
        0 if config["tie_word_embeddings"] else vocab_size * hidden_size
    )
    return (
        embedding_parameters
        + num_hidden_layers
        * (
            attention_parameters
            + query_key_norm_parameters
            + mlp_parameters
            + layer_norm_parameters
        )
        + final_norm_parameters
        + output_parameters
    )
