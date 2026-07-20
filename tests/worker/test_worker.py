"""Tests for Worker resource ownership and model execution."""

import pytest
import torch

from micro_vllm.config import CacheConfig
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.models.llama import LlamaModelConfig
from micro_vllm.worker.config import ModelExecutionConfig
from micro_vllm.worker.worker import Worker


def make_model_config() -> LlamaModelConfig:
    return LlamaModelConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )


def make_worker(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Worker:
    return Worker(
        make_model_config(),
        CacheConfig(block_size=2, num_blocks=8),
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


def test_worker_owns_configured_model_cache_and_runner() -> None:
    worker = make_worker()

    assert worker._model is not None
    assert worker._model.training is False
    assert next(worker._model.parameters()).device == torch.device("cpu")
    assert next(worker._model.parameters()).dtype is torch.float32
    assert worker._physical_kv_cache is not None
    assert worker._physical_kv_cache.device == torch.device("cpu")
    assert worker._physical_kv_cache.dtype is torch.float32
    assert worker._model_runner is not None


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
    physical_kv_cache = worker._physical_kv_cache
    assert physical_kv_cache is not None

    worker.shutdown()
    worker.shutdown()

    assert len(physical_kv_cache) == 0
    assert worker._model is None
    assert worker._physical_kv_cache is None
    assert worker._model_runner is None
    with pytest.raises(RuntimeError, match="worker is shut down"):
        worker.execute(make_plan())


def test_cpu_worker_executes_reference_model() -> None:
    worker = make_worker()

    output = worker.execute(make_plan())

    assert len(output.sampled_tokens) == 1
    assert output.sampled_tokens[0].request_id == "request-1"
    assert 0 <= output.sampled_tokens[0].token_id < make_model_config().vocab_size


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_worker_executes_reference_model() -> None:
    worker = make_worker(device="cuda", dtype=torch.float16)

    output = worker.execute(make_plan())

    assert len(output.sampled_tokens) == 1
    assert output.sampled_tokens[0].request_id == "request-1"
    assert 0 <= output.sampled_tokens[0].token_id < make_model_config().vocab_size
