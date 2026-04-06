"""Weight-only quantization — INT8/INT4 via group-wise scaling.

This is the LLM-style post-training quantization approach: weights are
compressed to low bit-width, activations stay in float. At inference,
weights are dequantized on the fly (or fused into a low-bit matmul
kernel). No calibration data is required for the basic "round-to-nearest"
variant; GPTQ/AWQ-style methods are planned as future variants.
"""

from __future__ import annotations

import copy
import time
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from comprexx.analysis.profiler import analyze
from comprexx.core.report import StageReport
from comprexx.stages.base import CompressionStage, StageContext


class WeightOnlyQuantConfig(BaseModel):
    """Configuration for weight-only quantization."""

    bits: Literal[4, 8] = 8
    group_size: int = Field(default=128, ge=8)
    symmetric: bool = True
    exclude_layers: list[str] = Field(default_factory=list)


def _quantize_tensor(
    w: torch.Tensor, bits: int, group_size: int, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Group-wise quantize a 2D weight matrix along the input dim.

    Returns (q_int, scale, zero_point). The returned `q_int` has the same
    dtype/shape as w (it holds the dequantized float values — this is the
    "simulated" quantization used for reporting and reference inference).
    """
    assert w.dim() == 2, "expected 2D weight"
    out_dim, in_dim = w.shape

    g = min(group_size, in_dim)
    pad = (g - in_dim % g) % g
    if pad:
        w_p = torch.nn.functional.pad(w, (0, pad))
    else:
        w_p = w
    groups = w_p.reshape(out_dim, -1, g)  # (out, n_groups, g)

    qmax = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1
    qmin = -(2 ** (bits - 1)) if symmetric else 0

    if symmetric:
        amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / qmax
        zero = None
        q = torch.round(groups / scale).clamp(qmin, qmax)
        dq = q * scale
    else:
        gmin = groups.amin(dim=-1, keepdim=True)
        gmax = groups.amax(dim=-1, keepdim=True)
        scale = ((gmax - gmin) / (qmax - qmin)).clamp(min=1e-8)
        zero = torch.round(-gmin / scale)
        q = torch.round(groups / scale + zero).clamp(qmin, qmax)
        dq = (q - zero) * scale

    dq = dq.reshape(out_dim, -1)
    if pad:
        dq = dq[:, :in_dim]
    return dq.to(w.dtype), scale, zero


class WeightOnlyQuant(CompressionStage):
    """Weight-only quantization stage (INT4/INT8, group-wise).

    Replaces Linear/Conv2d weights with their dequantized low-bit
    approximation. Size savings are reported as the theoretical packed
    footprint (bits * num_weights / 8 + scale overhead), not the on-disk
    float tensor size.
    """

    name = "weight_only_quant"

    def __init__(self, **kwargs):
        self.config = WeightOnlyQuantConfig(**kwargs)

    def _target_layers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        targets = []
        for name, module in model.named_modules():
            if name in self.config.exclude_layers:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                targets.append((name, module))
        return targets

    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        start = time.time()

        model = copy.deepcopy(model)
        model.eval()

        profile_before = analyze(model, context.input_shape, context.device)

        targets = self._target_layers(model)
        bits = self.config.bits
        gs = self.config.group_size
        notes: list[str] = []

        quantized_weights = 0
        total_quantized_bytes = 0
        for name, module in targets:
            w = module.weight.data
            orig_shape = w.shape
            w2d = w.reshape(orig_shape[0], -1)

            dq, scale, zero = _quantize_tensor(w2d, bits, gs, self.config.symmetric)
            module.weight.data = dq.reshape(orig_shape)

            n_weights = w2d.numel()
            quantized_weights += n_weights
            # Theoretical packed size: bits per weight + fp16 scales
            # (and fp16 zero-points for asymmetric)
            scale_count = scale.numel()
            per_scale_bytes = 2  # fp16
            total_quantized_bytes += (
                n_weights * bits // 8
                + scale_count * per_scale_bytes
                + (scale_count * per_scale_bytes if zero is not None else 0)
            )

        profile_after = analyze(model, context.input_shape, context.device)

        # Override size_bytes_after with the theoretical packed size for
        # quantized layers plus the unchanged size of everything else.
        untouched_bytes = profile_before.size_bytes
        for _, module in targets:
            # subtract the fp32 bytes for quantized weights
            untouched_bytes -= module.weight.numel() * 4
        theoretical_size_after = untouched_bytes + total_quantized_bytes

        if targets:
            notes.append(
                f"Quantized {quantized_weights:,} weights to INT{bits} "
                f"(group_size={gs}, symmetric={self.config.symmetric}) "
                f"across {len(targets)} layer(s)."
            )
            notes.append(
                f"Theoretical packed size: "
                f"{theoretical_size_after / 1024:.1f} KB "
                f"(vs {profile_before.size_bytes / 1024:.1f} KB dense fp32)."
            )
        else:
            notes.append("No eligible layers found.")

        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique=f"weight_only_int{bits}",
            duration_seconds=duration,
            param_count_before=profile_before.total_params,
            param_count_after=profile_after.total_params,
            flops_before=profile_before.total_flops,
            flops_after=profile_after.total_flops,
            size_bytes_before=profile_before.size_bytes,
            size_bytes_after=theoretical_size_after,
            notes=notes,
        )

        return model, report
