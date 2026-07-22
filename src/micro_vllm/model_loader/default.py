"""Default local and Hugging Face safetensors model loader."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from torch import nn

from micro_vllm.config import LLMConfig
from micro_vllm.model_loader.base import ModelLoader
from micro_vllm.model_loader.checkpoint import SafetensorsCheckpoint


_CHECKPOINT_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)


class DefaultModelLoader(ModelLoader):
    """Load local or Hugging Face models stored as safetensors."""

    def download_weights(self, config: LLMConfig) -> Path:
        local_path = _resolve_local_model_path(config.model)
        if local_path is not None:
            return local_path

        assert isinstance(config.model, str)
        return Path(
            snapshot_download(
                repo_id=config.model,
                revision=config.revision,
                cache_dir=config.cache_dir,
                token=config.token,
                local_files_only=config.local_files_only,
                allow_patterns=list(_CHECKPOINT_PATTERNS),
            )
        )

    def load_weights(self, model: nn.Module, model_path: Path) -> None:
        checkpoint = SafetensorsCheckpoint(model_path)
        model.load_weights(checkpoint.iter_tensors())


def _resolve_local_model_path(
    model_name_or_path: str | Path,
) -> Path | None:
    if isinstance(model_name_or_path, str):
        if not model_name_or_path.strip():
            raise ValueError("model name or path must be non-empty")
        model_path = Path(model_name_or_path).expanduser()
        if not model_path.exists():
            return None
    else:
        model_path = model_name_or_path.expanduser()

    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"model directory does not exist: {model_path}"
        )
    if not model_path.is_dir():
        raise ValueError(f"model path must be a directory: {model_path}")
    return model_path
