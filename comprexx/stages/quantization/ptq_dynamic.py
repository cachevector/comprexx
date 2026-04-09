"""Post-Training Dynamic Quantization (INT8)."""

from __future__ import annotations

import copy
import time
import warnings
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel

from comprexx.analysis.profiler import analyze
from comprexx.core.report import StageReport
from comprexx.stages.base import CompressionStage, StageContext

# TODO(v0.3): migrate to torchao.quantization. torch.ao.quantization is
# scheduled for removal in torch 2.10. Tracked at pytorch/ao#2259.
_TORCH_AO_WARN = r"torch\.ao\.quantization is deprecated"


class PTQDynamicConfig(BaseModel):
    """Configuration for dynamic quantization."""

    format: Literal["int8"] = "int8"


class PTQDynamic(CompressionStage):
    """Post-Training Dynamic Quantization.

    Quantizes weights statically and activations dynamically at runtime.
    No calibration data needed. Targets Linear and LSTM layers.
    """

    name = "ptq_dynamic"

    def __init__(self, **kwargs):
        self.config = PTQDynamicConfig(**kwargs)

    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        start = time.time()

        model = copy.deepcopy(model)
        model.eval()

        # Ensure quantization engine is available
        _set_qengine()

        # Profile before
        profile_before = analyze(model, context.input_shape, context.device)

        # Apply dynamic quantization. Silence the torch.ao.quantization
        # deprecation warning — it's noise for our users, and we track the
        # migration in the TODO above.
        dtype = torch.qint8
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=_TORCH_AO_WARN, category=DeprecationWarning
            )
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec={nn.Linear, nn.LSTM},
                dtype=dtype,
            )

        # Profile after — size calculation for quantized models
        size_after = _quantized_model_size(quantized_model)

        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique="ptq_dynamic_int8",
            duration_seconds=duration,
            param_count_before=profile_before.total_params,
            param_count_after=profile_before.total_params,  # param count unchanged
            flops_before=profile_before.total_flops,
            flops_after=profile_before.total_flops,  # FLOPs not reduced by dynamic quant
            size_bytes_before=profile_before.size_bytes,
            size_bytes_after=size_after,
            notes=["Dynamic quantization applied to Linear and LSTM layers."],
        )

        return quantized_model, report


def _set_qengine() -> None:
    """Set quantization engine to an available backend."""
    supported = torch.backends.quantized.supported_engines
    if "x86" in supported:
        torch.backends.quantized.engine = "x86"
    elif "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"


def _quantized_model_size(model: nn.Module) -> int:
    """Estimate model size in bytes, handling quantized parameters."""
    total = 0
    for param in model.parameters():
        total += param.nelement() * param.element_size()
    for buf in model.buffers():
        total += buf.nelement() * buf.element_size()

    # Account for quantized modules storing packed weights
    for module in model.modules():
        if hasattr(module, "_packed_params"):
            try:
                w, b = module._weight_bias()
                total += w.numel() * 1  # INT8 = 1 byte per element
                if b is not None:
                    total += b.nelement() * b.element_size()
            except Exception:
                pass
    return max(total, 1)  # Avoid zero
