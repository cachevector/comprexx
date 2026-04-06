"""Unstructured pruning — magnitude-based weight pruning."""

from __future__ import annotations

import copy
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pydantic import BaseModel, Field

from comprexx.analysis.profiler import analyze
from comprexx.core.report import StageReport
from comprexx.stages.base import CompressionStage, StageContext


class UnstructuredPruningConfig(BaseModel):
    """Configuration for unstructured (element-wise) pruning."""

    sparsity: float = Field(default=0.5, ge=0.0, le=1.0)
    criteria: Literal["magnitude", "random"] = "magnitude"
    scope: Literal["global", "local"] = "global"
    gradual_steps: int = Field(default=1, ge=1)
    exclude_layers: list[str] = Field(default_factory=list)


class UnstructuredPruning(CompressionStage):
    """Unstructured pruning: zeros individual weights by magnitude.

    Supports gradual pruning — the target sparsity is reached over N steps
    using a cubic schedule, which tends to preserve accuracy better than
    one-shot pruning at high sparsities.
    """

    name = "unstructured_pruning"

    def __init__(self, **kwargs):
        self.config = UnstructuredPruningConfig(**kwargs)

    def _get_target_layers(
        self, model: nn.Module
    ) -> list[tuple[str, nn.Module, str]]:
        """Find prunable layers (Conv2d, Linear) not in exclude list."""
        targets = []
        for name, module in model.named_modules():
            if name in self.config.exclude_layers:
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                targets.append((name, module, "weight"))
        return targets

    def _step_sparsity(self, step: int) -> float:
        """Cubic sparsity schedule for gradual pruning."""
        n = self.config.gradual_steps
        if n <= 1:
            return self.config.sparsity
        # Zhu & Gupta (2017) cubic schedule
        t = step / n
        return self.config.sparsity * (1.0 - (1.0 - t) ** 3)

    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        start = time.time()

        model = copy.deepcopy(model)
        model.eval()

        profile_before = analyze(model, context.input_shape, context.device)

        targets = self._get_target_layers(model)
        notes: list[str] = []

        if not targets:
            notes.append("No Conv2d/Linear layers found to prune.")
        else:
            for step in range(1, self.config.gradual_steps + 1):
                amount = self._step_sparsity(step)
                if self.config.scope == "global":
                    params = [(m, p) for _, m, p in targets]
                    method = (
                        prune.L1Unstructured
                        if self.config.criteria == "magnitude"
                        else prune.RandomUnstructured
                    )
                    prune.global_unstructured(
                        params, pruning_method=method, amount=amount
                    )
                else:
                    for _, module, pname in targets:
                        if self.config.criteria == "magnitude":
                            prune.l1_unstructured(module, pname, amount=amount)
                        else:
                            prune.random_unstructured(module, pname, amount=amount)

            # Make pruning permanent
            zeroed = 0
            total = 0
            for _, module, pname in targets:
                if hasattr(module, f"{pname}_mask"):
                    mask = getattr(module, f"{pname}_mask")
                    zeroed += int((mask == 0).sum().item())
                    total += int(mask.numel())
                    prune.remove(module, pname)

            actual = zeroed / total if total else 0.0
            notes.append(
                f"Pruned {zeroed}/{total} weights "
                f"({actual:.1%} actual sparsity) "
                f"across {len(targets)} layers "
                f"in {self.config.gradual_steps} step(s)."
            )

        profile_after = analyze(model, context.input_shape, context.device)

        # Unstructured pruning doesn't shrink tensors, so param count is
        # unchanged — compute effective sparsity instead.
        effective_zeros = 0
        total_weights = 0
        for _, module, pname in targets:
            w = getattr(module, pname).data
            effective_zeros += int((w == 0).sum().item())
            total_weights += int(w.numel())

        if total_weights:
            notes.append(
                f"Effective weight sparsity: "
                f"{effective_zeros / total_weights:.1%}"
            )

        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique=f"unstructured_pruning_{self.config.criteria}",
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
