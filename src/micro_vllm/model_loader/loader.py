"""Public entry point for the default model loader."""

from __future__ import annotations

from torch import nn

from micro_vllm.config import LLMConfig
from micro_vllm.model_loader.default import DefaultModelLoader


def load_model(config: LLMConfig) -> nn.Module:
    """Load one registered model from a local directory or Hugging Face Hub."""
    return DefaultModelLoader().load_model(config)
