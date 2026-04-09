"""Low-rank decomposition — SVD-based weight factorization.

Replaces a Linear layer W (out x in) with two smaller Linear layers
W = U @ V where U is (out x r), V is (r x in), and r << min(out, in).
This saves parameters and FLOPs whenever r * (out + in) < out * in.
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


class LowRankDecompositionConfig(BaseModel):
    """Configuration for SVD-based low-rank decomposition."""

    rank_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    energy_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    mode: Literal["ratio", "energy"] = "ratio"
    min_rank: int = Field(default=1, ge=1)
    exclude_layers: list[str] = Field(default_factory=list)


def _svd_factorize(
    weight: torch.Tensor, bias: torch.Tensor | None, rank: int
) -> tuple[nn.Linear, nn.Linear]:
    """Factorize a Linear weight into two Linear layers of inner dim `rank`."""
    out_dim, in_dim = weight.shape
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    # Keep top-`rank` singular components
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Fold sqrt(S) into both sides for numerical symmetry
    s_sqrt = S_r.sqrt()
    A = (Vh_r * s_sqrt.unsqueeze(-1))  # (rank, in_dim)
    B = (U_r * s_sqrt.unsqueeze(0))    # (out_dim, rank)

    first = nn.Linear(in_dim, rank, bias=False)
    first.weight.data = A

    second = nn.Linear(rank, out_dim, bias=bias is not None)
    second.weight.data = B
    if bias is not None:
        second.bias.data = bias.clone()

    return first, second


def _choose_rank(
    w: torch.Tensor, cfg: LowRankDecompositionConfig
) -> int:
    out_dim, in_dim = w.shape
    full = min(out_dim, in_dim)
    if cfg.mode == "energy" and cfg.energy_threshold is not None:
        S = torch.linalg.svdvals(w)
        energy = (S**2).cumsum(0) / (S**2).sum()
        r = int((energy >= cfg.energy_threshold).nonzero()[0].item()) + 1
    else:
        r = max(1, int(full * cfg.rank_ratio))
    return max(cfg.min_rank, min(r, full))


def _replace_module(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace a submodule by its dotted name."""
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


class LowRankDecomposition(CompressionStage):
    """Low-rank decomposition of Linear layers via truncated SVD.

    Only applies to layers where decomposition actually reduces parameters
    (rank * (out + in) < out * in). Other layers are left unchanged.
    """

    name = "low_rank_decomposition"

    def __init__(self, **kwargs):
        self.config = LowRankDecompositionConfig(**kwargs)

    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        start = time.time()

        model = copy.deepcopy(model)
        model.eval()

        profile_before = analyze(model, context.input_shape, context.device)

        # Collect target layers first (modifying modules during iteration
        # of named_modules is unsafe).
        targets: list[tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if name in self.config.exclude_layers:
                continue
            if isinstance(module, nn.Linear):
                targets.append((name, module))

        decomposed = 0
        skipped_no_gain = 0
        notes: list[str] = []

        for name, module in targets:
            w = module.weight.data
            out_dim, in_dim = w.shape
            rank = _choose_rank(w, self.config)

            # Only decompose if it actually saves parameters
            if rank * (out_dim + in_dim) >= out_dim * in_dim:
                skipped_no_gain += 1
                continue

            bias = module.bias.data if module.bias is not None else None
            first, second = _svd_factorize(w, bias, rank)
            replacement = nn.Sequential(first, second)
            _replace_module(model, name, replacement)
            decomposed += 1

        if decomposed:
            notes.append(
                f"Decomposed {decomposed} Linear layer(s) via truncated SVD "
                f"(mode={self.config.mode})."
            )
        if skipped_no_gain:
            notes.append(
                f"Skipped {skipped_no_gain} layer(s) where chosen rank "
                f"would not reduce parameters."
            )
        if not targets:
            notes.append("No Linear layers found to decompose.")

        profile_after = analyze(model, context.input_shape, context.device)
        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique="low_rank_svd",
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
