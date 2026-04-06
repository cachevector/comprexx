"""Operator fusion — folds Conv+BN (and Conv+BN+ReLU) into single ops.

Uses torch.fx symbolic tracing to find adjacent Conv2d+BatchNorm2d
patterns and replaces them with a single mathematically equivalent
Conv2d whose weights and bias absorb the BN affine transform. This
removes the BN layer from the forward pass at zero accuracy cost and
matches what inference engines do implicitly — exposing it here means
the savings show up in the report and downstream stages see a simpler
graph.
"""

from __future__ import annotations

import copy
import time

import torch
import torch.fx as fx
import torch.nn as nn
from pydantic import BaseModel, Field

from comprexx.analysis.profiler import analyze
from comprexx.core.report import StageReport
from comprexx.stages.base import CompressionStage, StageContext


class OperatorFusionConfig(BaseModel):
    """Configuration for operator fusion."""

    fuse_conv_bn: bool = True
    fallback_on_trace_error: bool = Field(default=True)


def _fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a new Conv2d equivalent to conv followed by bn in eval mode."""
    fused = copy.deepcopy(conv)

    w_conv = fused.weight.clone()
    if fused.bias is not None:
        b_conv = fused.bias.clone()
    else:
        b_conv = torch.zeros(w_conv.size(0), device=w_conv.device, dtype=w_conv.dtype)

    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_eps = bn.eps
    bn_w = bn.weight if bn.weight is not None else torch.ones_like(bn_mean)
    bn_b = bn.bias if bn.bias is not None else torch.zeros_like(bn_mean)

    inv_std = torch.rsqrt(bn_var + bn_eps)
    factor = bn_w * inv_std  # shape: (C,)

    # w_fused[c, :, :, :] = w_conv[c, :, :, :] * factor[c]
    w_fused = w_conv * factor.reshape(-1, 1, 1, 1)
    b_fused = (b_conv - bn_mean) * factor + bn_b

    new_conv = nn.Conv2d(
        fused.in_channels,
        fused.out_channels,
        kernel_size=fused.kernel_size,
        stride=fused.stride,
        padding=fused.padding,
        dilation=fused.dilation,
        groups=fused.groups,
        bias=True,
        padding_mode=fused.padding_mode,
    )
    new_conv.weight.data = w_fused
    new_conv.bias.data = b_fused
    return new_conv


def _fx_fuse(model: nn.Module) -> tuple[nn.Module, int]:
    """Use torch.fx to locate and fuse Conv2d+BatchNorm2d pairs."""
    traced: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(traced.named_modules())
    fused_count = 0

    patterns = []
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = modules.get(node.target)
        if not isinstance(module, nn.BatchNorm2d):
            continue
        prev = node.args[0] if node.args else None
        if not isinstance(prev, fx.Node) or prev.op != "call_module":
            continue
        prev_module = modules.get(prev.target)
        if not isinstance(prev_module, nn.Conv2d):
            continue
        # Only fuse if the conv's output has a single consumer (this bn)
        if len(prev.users) != 1:
            continue
        patterns.append((prev, node, prev_module, module))

    for conv_node, bn_node, conv_mod, bn_mod in patterns:
        fused_conv = _fuse_conv_bn_eval(conv_mod, bn_mod)
        # Install the fused conv in place of the original conv
        _set_submodule(traced, conv_node.target, fused_conv)
        # Redirect bn's users to conv_node
        bn_node.replace_all_uses_with(conv_node)
        traced.graph.erase_node(bn_node)
        # Remove BN submodule for cleanliness
        try:
            _delete_submodule(traced, bn_node.target)
        except AttributeError:
            pass
        fused_count += 1

    traced.graph.lint()
    traced.recompile()
    return traced, fused_count


def _set_submodule(root: nn.Module, qualified: str, new_module: nn.Module) -> None:
    parts = qualified.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _delete_submodule(root: nn.Module, qualified: str) -> None:
    parts = qualified.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    delattr(parent, parts[-1])


class OperatorFusion(CompressionStage):
    """Fuse adjacent Conv2d+BatchNorm2d into a single Conv2d using torch.fx."""

    name = "operator_fusion"

    def __init__(self, **kwargs):
        self.config = OperatorFusionConfig(**kwargs)

    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        start = time.time()

        model = copy.deepcopy(model)
        model.eval()

        profile_before = analyze(model, context.input_shape, context.device)
        notes: list[str] = []
        fused_model = model
        fused_count = 0

        if self.config.fuse_conv_bn:
            try:
                fused_model, fused_count = _fx_fuse(model)
            except Exception as e:
                if not self.config.fallback_on_trace_error:
                    raise
                notes.append(
                    f"torch.fx tracing failed ({type(e).__name__}); "
                    f"skipping fusion."
                )
                fused_model = model

        if fused_count:
            notes.append(f"Fused {fused_count} Conv2d+BatchNorm2d pair(s).")
        elif self.config.fuse_conv_bn and not notes:
            notes.append("No fusible Conv+BN patterns found.")

        profile_after = analyze(fused_model, context.input_shape, context.device)
        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique="operator_fusion_fx",
            duration_seconds=duration,
            param_count_before=profile_before.total_params,
            param_count_after=profile_after.total_params,
            flops_before=profile_before.total_flops,
            flops_after=profile_after.total_flops,
            size_bytes_before=profile_before.size_bytes,
            size_bytes_after=profile_after.size_bytes,
            notes=notes,
        )

        return fused_model, report
