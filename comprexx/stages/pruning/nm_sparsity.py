"""N:M sparsity — enforces N non-zero weights out of every M consecutive.

This pattern (notably 2:4) is accelerated natively by NVIDIA Ampere and
newer GPUs via sparse tensor cores, yielding real speedups at inference
time without the irregular access patterns of unstructured sparsity.
"""

from __future__ import annotations

import copy
import time
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator

from comprexx.analysis.profiler import analyze
from comprexx.core.report import StageReport
from comprexx.stages.base import CompressionStage, StageContext


class NMSparsityConfig(BaseModel):
    """Configuration for N:M sparsity.

    Keeps `n` non-zero weights out of every `m` consecutive weights along
    the input-channel dimension.
    """

    n: int = Field(default=2, ge=1)
    m: int = Field(default=4, ge=2)
    criteria: Literal["magnitude", "random"] = "magnitude"
    exclude_layers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_n_lt_m(self):
        if self.n >= self.m:
            raise ValueError(f"n ({self.n}) must be less than m ({self.m})")
        return self


def _apply_nm_mask(weight: torch.Tensor, n: int, m: int, random: bool) -> torch.Tensor:
    """Return a 0/1 mask enforcing N:M sparsity along the last dim.

    Weights are reshaped into groups of `m` along the flattened input
    dimension; within each group the top-`n` by magnitude (or random)
    survive.
    """
    orig_shape = weight.shape
    # Flatten everything except output dim: (out, in_total)
    w2d = weight.reshape(orig_shape[0], -1)
    out_dim, in_dim = w2d.shape

    # Pad so in_dim is divisible by m
    pad = (m - in_dim % m) % m
    if pad:
        w2d = torch.nn.functional.pad(w2d, (0, pad))

    groups = w2d.reshape(out_dim, -1, m)  # (out, n_groups, m)
    if random:
        scores = torch.rand_like(groups)
    else:
        scores = groups.abs()

    # Indices of top-n per group
    _, topk_idx = scores.topk(n, dim=-1)
    mask = torch.zeros_like(groups)
    mask.scatter_(-1, topk_idx, 1.0)

    mask = mask.reshape(out_dim, -1)
    if pad:
        mask = mask[:, :in_dim]
    return mask.reshape(orig_shape)


class NMSparsity(CompressionStage):
    """N:M structured sparsity stage."""

    name = "nm_sparsity"

    def __init__(self, **kwargs):
        self.config = NMSparsityConfig(**kwargs)

    def _target_layers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        targets = []
        for name, module in model.named_modules():
            if name in self.config.exclude_layers:
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
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
        notes: list[str] = []
        n, m = self.config.n, self.config.m
        random = self.config.criteria == "random"

        zeros, total = 0, 0
        skipped = 0
        for name, module in targets:
            w = module.weight.data
            flat_in = w.numel() // w.shape[0]
            if flat_in < m:
                skipped += 1
                continue
            mask = _apply_nm_mask(w, n, m, random)
            module.weight.data = w * mask
            zeros += int((mask == 0).sum().item())
            total += int(mask.numel())

        if skipped:
            notes.append(
                f"Skipped {skipped} layer(s) whose input dim was smaller than m={m}."
            )

        if total:
            notes.append(
                f"Applied {n}:{m} sparsity to "
                f"{len(targets) - skipped} layer(s) "
                f"({zeros / total:.1%} weights zeroed)."
            )
        else:
            notes.append("No layers eligible for N:M sparsity.")

        profile_after = analyze(model, context.input_shape, context.device)
        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique=f"nm_sparsity_{n}of{m}",
            duration_seconds=duration,
            param_count_before=profile_before.total_params,
            param_count_after=profile_after.total_params,
            flops_before=profile_before.total_flops,
            flops_after=profile_after.total_flops,
            size_bytes_before=profile_before.size_bytes,
            size_bytes_after=profile_after.size_bytes,
            notes=notes,
        )

        return model, report
