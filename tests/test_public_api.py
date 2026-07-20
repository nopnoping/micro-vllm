"""Tests for the intentionally small public package surface."""

from inspect import getdoc

import micro_vllm
import micro_vllm.core as core
import micro_vllm.engine as engine
import micro_vllm.executor as executor
import micro_vllm.models as models
import micro_vllm.sampling as sampling
import micro_vllm.worker as worker
from micro_vllm.config import CacheConfig, SchedulerConfig
from micro_vllm.core.request import FinishReason, Request, RequestStatus
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.engine.output import RequestOutput
from micro_vllm.executor.interface import Executor, ModelRunnerOutput, SampledToken
from micro_vllm.models.llama import LlamaModelConfig
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.config import ModelExecutionConfig
from micro_vllm.worker.kv_cache import PhysicalKVCache


def test_top_level_exports_engine_entry_points() -> None:
    assert micro_vllm.__all__ == (
        "CacheConfig",
        "Engine",
        "RequestOutput",
        "SamplingParams",
        "SchedulerConfig",
    )
    assert micro_vllm.CacheConfig is CacheConfig
    assert micro_vllm.Engine is Engine
    assert micro_vllm.RequestOutput is RequestOutput
    assert micro_vllm.SamplingParams is SamplingParams
    assert micro_vllm.SchedulerConfig is SchedulerConfig


def test_core_exports_only_stable_domain_and_scheduling_types() -> None:
    assert core.__all__ == (
        "FinishReason",
        "Request",
        "RequestStatus",
        "ScheduledRequest",
        "Scheduler",
        "SchedulerOutput",
    )
    assert core.FinishReason is FinishReason
    assert core.Request is Request
    assert core.RequestStatus is RequestStatus
    assert core.ScheduledRequest is ScheduledRequest
    assert core.Scheduler is Scheduler
    assert core.SchedulerOutput is SchedulerOutput


def test_engine_executor_and_sampling_exports_are_explicit() -> None:
    assert engine.__all__ == ("Engine", "RequestOutput")
    assert executor.__all__ == ("Executor", "ModelRunnerOutput", "SampledToken")
    assert sampling.__all__ == ("SamplingParams",)

    assert engine.Engine is Engine
    assert engine.RequestOutput is RequestOutput
    assert executor.Executor is Executor
    assert executor.ModelRunnerOutput is ModelRunnerOutput
    assert executor.SampledToken is SampledToken
    assert sampling.SamplingParams is SamplingParams


def test_model_and_worker_configuration_exports_are_explicit() -> None:
    assert models.__all__ == ("LlamaModelConfig",)
    assert worker.__all__ == ("ModelExecutionConfig", "PhysicalKVCache")
    assert models.LlamaModelConfig is LlamaModelConfig
    assert worker.ModelExecutionConfig is ModelExecutionConfig
    assert worker.PhysicalKVCache is PhysicalKVCache


def test_fake_executor_is_not_public() -> None:
    for package in (micro_vllm, core, engine, executor, sampling):
        assert "FakeExecutor" not in package.__all__
        assert not hasattr(package, "FakeExecutor")


def test_primary_public_contracts_have_documentation() -> None:
    for public_contract in (Request, SchedulerOutput, Executor, Engine):
        assert getdoc(public_contract)
