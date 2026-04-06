"""Weight clustering — shares weights via k-means quantization.

Each layer's weights are clustered into `k` centroids and every weight
is replaced with its nearest centroid. This is effectively a form of
scalar quantization to a learned (non-uniform) codebook. The model
runs in fp32 with only `k` unique weight values per layer, which gives
substantial compression when serialized with an indexed representation
(bits per weight = ceil(log2(k)) + codebook overhead).
"""

from __future__ import annotations

import copy
import math
import time
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from comprexx.analysis.profiler import analyze
from comprexx.core.report import StageReport
from comprexx.stages.base import CompressionStage, StageContext


class WeightClusteringConfig(BaseModel):
    """Configuration for weight clustering."""

    num_clusters: int = Field(default=16, ge=2, le=65536)
    init: Literal["linear", "random", "density"] = "linear"
    max_iter: int = Field(default=15, ge=1)
    per_layer: bool = True
    exclude_layers: list[str] = Field(default_factory=list)


def _init_centroids(
    values: torch.Tensor, k: int, method: str
) -> torch.Tensor:
    """Initialize k centroids from a 1-D tensor of weight values."""
    v_min = values.min().item()
    v_max = values.max().item()
    if v_min == v_max:
        return torch.full((k,), v_min, dtype=values.dtype, device=values.device)
    if method == "linear":
        return torch.linspace(v_min, v_max, k, dtype=values.dtype, device=values.device)
    if method == "random":
        idx = torch.randint(0, values.numel(), (k,), device=values.device)
        return values[idx].clone()
    if method == "density":
        # Pick k quantiles of the weight distribution
        qs = torch.linspace(0, 1, k, device=values.device)
        return torch.quantile(values, qs).to(values.dtype)
    raise ValueError(f"Unknown init method: {method}")


def _kmeans_1d(
    values: torch.Tensor, k: int, init: str, max_iter: int
) -> torch.Tensor:
    """Run 1-D k-means. Returns the quantized values (same shape as input)."""
    flat = values.flatten()
    n = flat.numel()
    if n == 0:
        return values
    k = min(k, n)
    centroids = _init_centroids(flat, k, init)

    for _ in range(max_iter):
        # Assign each weight to nearest centroid
        dists = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        # Update centroids to cluster means
        new_centroids = centroids.clone()
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = flat[mask].mean()
        if torch.allclose(new_centroids, centroids, atol=1e-8):
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final assignment
    dists = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    assignments = dists.argmin(dim=1)
    quantized = centroids[assignments]
    return quantized.reshape(values.shape)


class WeightClustering(CompressionStage):
    """Per-layer k-means weight clustering."""

    name = "weight_clustering"

    def __init__(self, **kwargs):
        self.config = WeightClusteringConfig(**kwargs)

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
        k = self.config.num_clusters
        notes: list[str] = []

        total_weights = 0
        clustered_layers = 0
        for name, module in targets:
            w = module.weight.data
            q = _kmeans_1d(w, k, self.config.init, self.config.max_iter)
            module.weight.data = q
            total_weights += w.numel()
            clustered_layers += 1

        if clustered_layers:
            bits_per_idx = max(1, math.ceil(math.log2(k)))
            # Theoretical packed size: bits_per_idx * num_weights + codebook
            # (codebook = k * fp32 per layer).
            codebook_bytes = clustered_layers * k * 4
            packed_weight_bytes = (total_weights * bits_per_idx + 7) // 8
            non_clustered_bytes = profile_before.size_bytes - total_weights * 4
            theoretical = non_clustered_bytes + packed_weight_bytes + codebook_bytes
            notes.append(
                f"Clustered {total_weights:,} weights across "
                f"{clustered_layers} layer(s) to k={k} centroids."
            )
            notes.append(
                f"Theoretical packed size: "
                f"{theoretical / 1024:.1f} KB "
                f"(vs {profile_before.size_bytes / 1024:.1f} KB dense fp32)."
            )
        else:
            notes.append("No layers eligible for clustering.")
            theoretical = profile_before.size_bytes

        profile_after = analyze(model, context.input_shape, context.device)
        duration = time.time() - start

        report = StageReport(
            stage_name=self.name,
            technique=f"weight_clustering_k{k}",
            duration_seconds=duration,
            param_count_before=profile_before.total_params,
            param_count_after=profile_after.total_params,
            flops_before=profile_before.total_flops,
            flops_after=profile_after.total_flops,
            size_bytes_before=profile_before.size_bytes,
            size_bytes_after=theoretical,
            notes=notes,
        )

        return model, report
