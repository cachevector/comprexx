"""Model profiling: analyze architecture, parameters, FLOPs, and compressibility."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch.nn as nn

from comprexx.analysis.flops import count_flops

# Layer types considered compressible
_COMPRESSIBLE_TYPES = (nn.Conv2d, nn.Conv1d, nn.Linear)

# Layer types used for architecture detection
_CNN_TYPES = (nn.Conv2d, nn.Conv1d)
_TRANSFORMER_TYPES = (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
_RNN_TYPES = (nn.LSTM, nn.GRU, nn.RNN)


@dataclass
class LayerInfo:
    """Information about a single layer in a model."""

    name: str
    layer_type: str
    param_count: int
    flops: int
    is_compressible: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelProfile:
    """Complete profile of a model's structure and compute characteristics."""

    model_name: str
    total_params: int
    trainable_params: int
    total_flops: int
    size_bytes: int
    architecture_category: str  # "cnn", "transformer", "hybrid", "rnn", "unknown"
    layers: list[LayerInfo] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def compressible_layers(self) -> list[LayerInfo]:
        return [layer for layer in self.layers if layer.is_compressible]

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"  Architecture: {self.architecture_category}",
            f"  Parameters:   {self.total_params:,} ({self.total_params / 1e6:.1f}M)",
            f"  Trainable:    {self.trainable_params:,}",
            f"  FLOPs:        {self.total_flops:,} ({self.total_flops / 1e9:.2f} GFLOPs)",
            f"  Size:         {self.size_mb:.2f} MB",
            f"  Layers:       {len(self.layers)} "
            f"({len(self.compressible_layers())} compressible)",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "total_flops": self.total_flops,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_mb,
            "architecture_category": self.architecture_category,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def _detect_architecture(model: nn.Module) -> str:
    """Detect model architecture category by scanning module types."""
    has_cnn = False
    has_transformer = False
    has_rnn = False

    for module in model.modules():
        if isinstance(module, _CNN_TYPES):
            has_cnn = True
        elif isinstance(module, _TRANSFORMER_TYPES):
            has_transformer = True
        elif isinstance(module, _RNN_TYPES):
            has_rnn = True

    if has_cnn and has_transformer:
        return "hybrid"
    if has_transformer:
        return "transformer"
    if has_cnn:
        return "cnn"
    if has_rnn:
        return "rnn"
    return "unknown"


def _compute_model_size(model: nn.Module) -> int:
    """Compute model size in bytes from parameters."""
    total = 0
    for param in model.parameters():
        total += param.nelement() * param.element_size()
    for buf in model.buffers():
        total += buf.nelement() * buf.element_size()
    return total


def analyze(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cpu",
    model_name: Optional[str] = None,
) -> ModelProfile:
    """Analyze a model and produce a ModelProfile.

    Args:
        model: PyTorch nn.Module to analyze.
        input_shape: Input tensor shape including batch dimension.
        device: Device to run analysis on.
        model_name: Optional name. Defaults to class name.

    Returns:
        ModelProfile with full analysis results.
    """
    if model_name is None:
        model_name = model.__class__.__name__

    model = model.to(device)
    model.eval()

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs
    total_flops, per_layer_flops = count_flops(model, input_shape, device)

    # Size
    size_bytes = _compute_model_size(model)

    # Architecture
    architecture = _detect_architecture(model)

    # Layer info
    layers = []
    for name, module in model.named_modules():
        # Skip container modules (Sequential, ModuleList, etc.)
        if len(list(module.children())) > 0:
            continue
        if not name:
            continue

        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        flops = per_layer_flops.get(name, 0)
        is_compressible = isinstance(module, _COMPRESSIBLE_TYPES)

        layers.append(
            LayerInfo(
                name=name,
                layer_type=type(module).__name__,
                param_count=param_count,
                flops=flops,
                is_compressible=is_compressible,
            )
        )

    return ModelProfile(
        model_name=model_name,
        total_params=total_params,
        trainable_params=trainable_params,
        total_flops=total_flops,
        size_bytes=size_bytes,
        architecture_category=architecture,
        layers=layers,
    )
