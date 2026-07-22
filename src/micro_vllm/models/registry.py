"""Static mapping from external architecture names to model classes."""

from __future__ import annotations

from types import MappingProxyType

from torch import nn

from micro_vllm.config import ModelConfig
from micro_vllm.models.llama import LlamaModel
from micro_vllm.models.mistral import MistralModel
from micro_vllm.models.qwen import Qwen3Model


_MODEL_CLASSES = MappingProxyType(
    {
        "LlamaForCausalLM": LlamaModel,
        "MistralForCausalLM": MistralModel,
        "Qwen3ForCausalLM": Qwen3Model,
    }
)


def get_model_class(
    config: ModelConfig,
    *,
    source_label: str,
) -> type[nn.Module]:
    """Return the first registered architecture requested by the config."""
    for architecture in config.architectures:
        model_class = _MODEL_CLASSES.get(architecture)
        if model_class is not None:
            return model_class

    raise ValueError(
        f"unsupported model architectures {list(config.architectures)!r}; "
        f"supported architectures: {list(_MODEL_CLASSES)!r}; "
        f"source: {source_label}"
    )
