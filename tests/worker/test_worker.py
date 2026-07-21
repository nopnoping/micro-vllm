"""Tests for Worker resource ownership and model execution."""

import pytest
import torch

from micro_vllm.config import ModelConfig, SchedulerConfig
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.models.llama import LlamaModel
from micro_vllm.worker.config import (
    KVCacheInitializationConfig,
    ModelExecutionConfig,
)
from micro_vllm.worker.worker import Worker


def make_model_config() -> ModelConfig:
    return ModelConfig(
        {
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
        }
    )


def make_worker(
    *,
    model: LlamaModel | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Worker:
    if model is None:
        model = LlamaModel(make_model_config(), ReferenceAttentionBackend())
    return Worker(
        lambda: model,
        KVCacheInitializationConfig(block_size=2, max_num_blocks=8),
        SchedulerConfig(
            max_num_batched_tokens=8,
            max_num_seqs=2,
            max_model_len=32,
        ),
        ModelExecutionConfig(device=device, dtype=dtype),
    )


def make_plan() -> SchedulerOutput:
    return SchedulerOutput(
        (
            ScheduledRequest(
                request_id="request-1",
                input_token_ids=(5, 6),
                num_computed_tokens=0,
                num_scheduled_tokens=2,
                requires_sampling=True,
                block_table=(3,),
            ),
        )
    )


def test_worker_orchestrates_model_runner_lifecycle() -> None:
    model = LlamaModel(make_model_config(), ReferenceAttentionBackend())
    model.train()
    worker = make_worker(model=model)

    assert not hasattr(worker, "_model")
    assert worker._model_runner is not None
    assert worker._model_runner._model is model
    assert worker._model_runner._available_kv_cache_memory_bytes is None
    assert model.training is False
    assert next(model.parameters()).device == torch.device("cpu")
    assert next(model.parameters()).dtype is torch.float32
    assert not hasattr(worker, "_physical_kv_cache")
    physical_kv_cache = worker._model_runner._physical_kv_cache
    assert physical_kv_cache is not None
    assert physical_kv_cache.shape == torch.Size((2, 8, 2, 1, 4))
    assert physical_kv_cache.device == torch.device("cpu")
    assert physical_kv_cache.dtype is torch.float32
    assert worker.cache_config.num_blocks == 8


def test_worker_returns_model_runner_output_unchanged() -> None:
    class RecordingRunner:
        def __init__(self, output: ModelRunnerOutput) -> None:
            self.output = output
            self.plans: list[SchedulerOutput] = []

        def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
            self.plans.append(plan)
            return self.output

    worker = make_worker()
    output = ModelRunnerOutput((SampledToken("request-1", 8),))
    runner = RecordingRunner(output)
    worker._model_runner = runner
    plan = make_plan()

    result = worker.execute(plan)

    assert runner.plans == [plan]
    assert runner.plans[0] is plan
    assert result is output


def test_shutdown_releases_resources_and_execute_fails() -> None:
    worker = make_worker()
    model_runner = worker._model_runner
    assert model_runner is not None
    physical_kv_cache = model_runner._physical_kv_cache
    assert physical_kv_cache is not None

    worker.shutdown()
    worker.shutdown()

    assert len(physical_kv_cache) == 0
    assert model_runner._model is None
    assert model_runner._physical_kv_cache is None
    assert worker._model_runner is None
    with pytest.raises(RuntimeError, match="worker is shut down"):
        worker.execute(make_plan())


def test_cpu_worker_executes_reference_model() -> None:
    worker = make_worker()

    output = worker.execute(make_plan())

    assert len(output.sampled_tokens) == 1
    assert output.sampled_tokens[0].request_id == "request-1"
    assert (
        0
        <= output.sampled_tokens[0].token_id
        < make_model_config().require_positive_int("vocab_size")
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_worker_executes_reference_model() -> None:
    model = LlamaModel(make_model_config(), ReferenceAttentionBackend())
    worker = make_worker(model=model, device="cuda", dtype=torch.float16)

    output = worker.execute(make_plan())

    assert worker._model_runner is not None
    assert worker._model_runner._model is model
    assert worker._model_runner._available_kv_cache_memory_bytes is not None
    assert worker._model_runner._available_kv_cache_memory_bytes > 0
    assert worker._model_runner._peak_activation_memory_bytes >= 0
    assert worker.cache_config.num_blocks == 8
    assert next(model.parameters()).device.type == "cuda"
    assert next(model.parameters()).dtype is torch.float16
    assert len(output.sampled_tokens) == 1
    assert output.sampled_tokens[0].request_id == "request-1"
    assert (
        0
        <= output.sampled_tokens[0].token_id
        < make_model_config().require_positive_int("vocab_size")
    )
