"""FLOPs / MACs counter using forward hooks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _conv_flops(module: nn.Conv2d, input: torch.Tensor, output: torch.Tensor) -> int:
    batch_size = output.shape[0]
    out_h, out_w = output.shape[2], output.shape[3]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
    # 2 ops per multiply-add
    flops = 2 * batch_size * module.out_channels * out_h * out_w * kernel_ops
    if module.bias is not None:
        flops += batch_size * module.out_channels * out_h * out_w
    return flops


def _linear_flops(module: nn.Linear, input: torch.Tensor, output: torch.Tensor) -> int:
    batch_size = input.shape[0]
    # 2 ops per multiply-add
    flops = 2 * batch_size * module.in_features * module.out_features
    if module.bias is not None:
        flops += batch_size * module.out_features
    return flops


def _bn_flops(module: nn.BatchNorm2d, input: torch.Tensor, output: torch.Tensor) -> int:
    # BN: 2 ops per element (subtract mean, divide by std) + 2 for affine
    return 4 * input.numel()


_FLOP_HANDLERS: dict[type, Any] = {
    nn.Conv2d: _conv_flops,
    nn.Conv1d: lambda m, i, o: 2 * o.shape[0] * m.out_channels * o.shape[2] * m.kernel_size[0] * (m.in_channels // m.groups),
    nn.Linear: _linear_flops,
    nn.BatchNorm2d: _bn_flops,
    nn.BatchNorm1d: lambda m, i, o: 4 * i.numel(),
}


def count_flops(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cpu",
) -> tuple[int, dict[str, int]]:
    """Count FLOPs for a model given an input shape.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (including batch dim).
        device: Device to run the forward pass on.

    Returns:
        Tuple of (total_flops, per_layer_flops_dict).
    """
    model = model.to(device)
    model.eval()

    flops_dict: dict[str, int] = {}
    hooks = []

    def make_hook(name: str, handler):
        def hook_fn(module, inp, out):
            inp_tensor = inp[0] if isinstance(inp, tuple) else inp
            flops_dict[name] = handler(module, inp_tensor, out)
        return hook_fn

    for name, module in model.named_modules():
        for mod_type, handler in _FLOP_HANDLERS.items():
            if isinstance(module, mod_type):
                hooks.append(module.register_forward_hook(make_hook(name, handler)))
                break

    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    total_flops = sum(flops_dict.values())
    return total_flops, flops_dict
