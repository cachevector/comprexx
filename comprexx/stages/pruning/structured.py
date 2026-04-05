"""Structured pruning — filter/channel pruning for Conv2d layers."""

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


class StructuredPruningConfig(BaseModel):
    """Configuration for structured pruning."""

    sparsity: float = Field(default=0.3, ge=0.0, le=1.0)
    criteria: Literal["l1_norm", "l2_norm", "random"] = "l1_norm"
    scope: Literal["global", "local"] = "global"
    target: Literal["filters", "channels"] = "filters"
    exclude_layers: list[str] = Field(default_factory=list)


class StructuredPruning(CompressionStage):
    """Structured pruning stage: removes entire filters/channels from Conv2d layers."""

    name = "structured_pruning"

    def __init__(self, **kwargs):
        self.config = StructuredPruningConfig(**kwargs)

    def _get_target_layers(self, model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
        """Find all Conv2d layers not in exclude list."""
        targets = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name not in self.config.exclude_layers:
                targets.append((name, module))
        return targets

    def _compute_importance(self, module: nn.Conv2d) -> torch.Tensor:
        """Compute per-filter importance scores."""
        weight = module.weight.data  # shape: (out_channels, in_channels, H, W)

        if self.config.criteria == "l1_norm":
            return weight.abs().sum(dim=(1, 2, 3))
        elif self.config.criteria == "l2_norm":
            return weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
        else:  # random
            return torch.rand(weight.shape[0])

    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        start = time.time()

        model = copy.deepcopy(model)
        model.eval()

        # Profile before
        profile_before = analyze(model, context.input_shape, context.device)

        targets = self._get_target_layers(model)
        notes = []

        if not targets:
            notes.append("No Conv2d layers found to prune.")
        elif self.config.scope == "global":
            self._prune_global(model, targets, notes)
        else:
            self._prune_local(targets, notes)

        # Make pruning permanent
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, "weight_mask"):
                prune.remove(module, "weight")

        # Profile after
        profile_after = analyze(model, context.input_shape, context.device)

        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique=f"structured_pruning_{self.config.criteria}",
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

    def _prune_global(
        self,
        model: nn.Module,
        targets: list[tuple[str, nn.Conv2d]],
        notes: list[str],
    ) -> None:
        """Apply global structured pruning across all target layers."""
        # Collect all filter importance scores
        all_scores = []
        for name, module in targets:
            scores = self._compute_importance(module)
            for i, s in enumerate(scores):
                all_scores.append((s.item(), name, module, i))

        # Sort by importance (ascending) and determine cutoff
        all_scores.sort(key=lambda x: x[0])
        n_prune = int(len(all_scores) * self.config.sparsity)

        if n_prune == 0:
            notes.append("Sparsity too low to prune any filters.")
            return

        # Group filters to prune by module
        to_prune: dict[str, set[int]] = {}
        for _, name, module, idx in all_scores[:n_prune]:
            to_prune.setdefault(name, set()).add(idx)

        # Apply pruning via masks
        for name, module in targets:
            if name in to_prune:
                n_filters = module.out_channels
                # Don't prune all filters — keep at least 1
                prune_indices = to_prune[name]
                if len(prune_indices) >= n_filters:
                    prune_indices = set(list(prune_indices)[: n_filters - 1])

                mask = torch.ones(n_filters, device=module.weight.device)
                for idx in prune_indices:
                    mask[idx] = 0.0

                # Apply structured mask along dim 0 (output channels / filters)
                prune.custom_from_mask(module, "weight", mask.view(-1, 1, 1, 1).expand_as(module.weight))

        notes.append(f"Global pruning: zeroed {n_prune} filters across {len(to_prune)} layers.")

    def _prune_local(
        self,
        targets: list[tuple[str, nn.Conv2d]],
        notes: list[str],
    ) -> None:
        """Apply per-layer structured pruning."""
        total_pruned = 0
        for name, module in targets:
            n_filters = module.out_channels
            n_prune = max(1, int(n_filters * self.config.sparsity))
            # Keep at least 1 filter
            n_prune = min(n_prune, n_filters - 1)

            if self.config.criteria == "random":
                prune.random_structured(module, "weight", amount=self.config.sparsity, dim=0)
            else:
                norm_type = 1 if self.config.criteria == "l1_norm" else 2
                prune.ln_structured(
                    module, "weight", amount=self.config.sparsity, n=norm_type, dim=0
                )
            total_pruned += n_prune

        notes.append(
            f"Local pruning: ~{total_pruned} filters pruned across {len(targets)} layers "
            f"at {self.config.sparsity:.0%} sparsity."
        )
