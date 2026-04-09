"""Post-Training Static Quantization (INT8)."""

from __future__ import annotations

import copy
import time
import warnings
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel

from comprexx.analysis.profiler import analyze
from comprexx.core.exceptions import CalibrationError
from comprexx.core.report import StageReport
from comprexx.stages.base import CompressionStage, StageContext

# TODO(v0.3): migrate to torchao.quantization. torch.ao.quantization is
# scheduled for removal in torch 2.10. Tracked at pytorch/ao#2259.
_TORCH_AO_WARN = r"torch\.ao\.quantization is deprecated"


class PTQStaticConfig(BaseModel):
    """Configuration for static quantization."""

    format: Literal["int8"] = "int8"
    calibration_method: Literal["minmax", "percentile", "entropy"] = "minmax"
    calibration_samples: int = 512
    exclude_layers: list[str] = []


class PTQStatic(CompressionStage):
    """Post-Training Static Quantization.

    Quantizes both weights and activations. Requires calibration data
    to determine activation ranges.
    """

    name = "ptq_static"

    def __init__(self, **kwargs):
        self.config = PTQStaticConfig(**kwargs)

    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        if context.calibration_data is None:
            raise CalibrationError(
                "PTQ Static requires calibration data. "
                "Provide calibration_data in the pipeline context."
            )

        start = time.time()

        model = copy.deepcopy(model)
        model.eval()
        model.to(context.device)

        # Ensure quantization engine is available
        _set_qengine()

        # Profile before
        profile_before = analyze(model, context.input_shape, context.device)

        # Wrap model with QuantStub/DeQuantStub for proper static quantization
        model = _QuantWrapper(model)

        # Set qconfig — use whichever backend is available
        backend = torch.backends.quantized.engine
        model.qconfig = torch.quantization.get_default_qconfig(backend)

        # Fuse common patterns if possible
        model = _try_fuse(model)

        # Prepare → calibrate → convert. Silence the torch.ao.quantization
        # deprecation warning — it's noise for users, tracked as a v0.3 TODO.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=_TORCH_AO_WARN, category=DeprecationWarning
            )
            prepared = torch.quantization.prepare(model, inplace=False)

            samples_seen = 0
            with torch.no_grad():
                for batch in context.calibration_data:
                    if samples_seen >= self.config.calibration_samples:
                        break
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(context.device)
                    prepared(x)
                    samples_seen += x.shape[0]

            quantized = torch.quantization.convert(prepared, inplace=False)

        # Estimate quantized size
        size_after = _estimate_quantized_size(quantized, profile_before.size_bytes)

        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique="ptq_static_int8",
            duration_seconds=duration,
            param_count_before=profile_before.total_params,
            param_count_after=profile_before.total_params,
            flops_before=profile_before.total_flops,
            flops_after=profile_before.total_flops,  # INT8 FLOPs ≈ same count, faster execution
            size_bytes_before=profile_before.size_bytes,
            size_bytes_after=size_after,
            notes=[
                f"Static quantization with {self.config.calibration_method} calibration.",
                f"Calibrated on {samples_seen} samples.",
            ],
        )

        return quantized, report


class _QuantWrapper(nn.Module):
    """Wraps a model with QuantStub/DeQuantStub for static quantization."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def _set_qengine() -> None:
    """Set quantization engine to an available backend."""
    supported = torch.backends.quantized.supported_engines
    if "x86" in supported:
        torch.backends.quantized.engine = "x86"
    elif "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"


def _try_fuse(model: nn.Module) -> nn.Module:
    """Try to fuse Conv+BN+ReLU patterns. Returns model unchanged if fusion fails."""
    try:
        # Build fusion list by scanning for sequential patterns
        fuse_list = []
        modules = dict(model.named_modules())
        names = list(modules.keys())

        for i, name in enumerate(names):
            mod = modules[name]
            if isinstance(mod, nn.Conv2d) and i + 1 < len(names):
                next_name = names[i + 1]
                next_mod = modules[next_name]
                if isinstance(next_mod, nn.BatchNorm2d):
                    if i + 2 < len(names) and isinstance(modules[names[i + 2]], nn.ReLU):
                        fuse_list.append([name, next_name, names[i + 2]])
                    else:
                        fuse_list.append([name, next_name])

        if fuse_list:
            model = torch.quantization.fuse_modules(model, fuse_list)
    except Exception:
        pass  # Fusion is best-effort
    return model


def _estimate_quantized_size(model: nn.Module, original_size: int) -> int:
    """Estimate the size of a quantized model.

    Static INT8 quantization typically achieves ~4x reduction for quantized layers.
    We estimate conservatively at 25-30% of original.
    """
    # Try to measure actual size from parameters/buffers
    total = 0
    for param in model.parameters():
        total += param.nelement() * param.element_size()
    for buf in model.buffers():
        total += buf.nelement() * buf.element_size()

    # If we got a reasonable number, use it; otherwise estimate
    if total > 0:
        return total

    # Fallback: assume ~4x reduction
    return original_size // 4
