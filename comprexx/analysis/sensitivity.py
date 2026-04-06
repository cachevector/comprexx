"""Sensitivity analysis — per-layer accuracy-impact estimation.

For each eligible layer, the analyzer temporarily applies a lightweight
perturbation (prune-by-magnitude or add Gaussian noise) at a fixed
intensity, re-runs the user's eval function, and records how much the
chosen metric dropped. The result is a ranking of which layers are
"sensitive" — compressing them is risky — and which are "tolerant" —
safe to compress aggressively.

This gives recipe authors a principled way to populate `exclude_layers`
and per-layer sparsity targets instead of picking them by hand.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal

import torch
import torch.nn as nn


@dataclass
class LayerSensitivity:
    """Sensitivity result for a single layer."""

    name: str
    layer_type: str
    baseline_metric: float
    perturbed_metric: float
    metric_drop: float
    num_params: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SensitivityReport:
    """Full sensitivity analysis result across all analyzed layers."""

    metric_name: str
    perturbation: str
    intensity: float
    baseline_metric: float
    layers: list[LayerSensitivity] = field(default_factory=list)

    def most_sensitive(self, n: int = 5) -> list[LayerSensitivity]:
        """Layers with the largest metric drops (in descending order)."""
        return sorted(self.layers, key=lambda l: l.metric_drop, reverse=True)[:n]

    def most_tolerant(self, n: int = 5) -> list[LayerSensitivity]:
        """Layers with the smallest metric drops."""
        return sorted(self.layers, key=lambda l: l.metric_drop)[:n]

    def recommend_exclusions(self, threshold: float) -> list[str]:
        """Names of layers whose drop exceeds `threshold` — candidates for
        `exclude_layers` in a pruning/quantization stage."""
        return [l.name for l in self.layers if l.metric_drop > threshold]

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "perturbation": self.perturbation,
            "intensity": self.intensity,
            "baseline_metric": self.baseline_metric,
            "layers": [l.to_dict() for l in self.layers],
        }

    def summary(self) -> str:
        lines = [
            f"Sensitivity analysis ({self.perturbation} @ {self.intensity})",
            f"  baseline {self.metric_name} = {self.baseline_metric:.4f}",
            f"  {len(self.layers)} layer(s) analyzed",
            "  most sensitive:",
        ]
        for l in self.most_sensitive(5):
            lines.append(
                f"    {l.name:40s}  drop={l.metric_drop:+.4f}  params={l.num_params:,}"
            )
        lines.append("  most tolerant:")
        for l in self.most_tolerant(5):
            lines.append(
                f"    {l.name:40s}  drop={l.metric_drop:+.4f}  params={l.num_params:,}"
            )
        return "\n".join(lines)


def _perturb_prune(weight: torch.Tensor, intensity: float) -> torch.Tensor:
    """Zero out the smallest-magnitude `intensity` fraction of weights."""
    k = int(weight.numel() * intensity)
    if k == 0:
        return weight.clone()
    flat = weight.abs().flatten()
    threshold = flat.kthvalue(k).values
    mask = (weight.abs() > threshold).to(weight.dtype)
    return weight * mask


def _perturb_noise(weight: torch.Tensor, intensity: float) -> torch.Tensor:
    """Add Gaussian noise scaled by the weight tensor's std."""
    std = weight.std().clamp(min=1e-8)
    return weight + torch.randn_like(weight) * std * intensity


_PERTURBATIONS = {
    "prune": _perturb_prune,
    "noise": _perturb_noise,
}


def analyze_sensitivity(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], dict[str, float]],
    metric: str = "top1_accuracy",
    perturbation: Literal["prune", "noise"] = "prune",
    intensity: float = 0.3,
    layer_types: tuple[type, ...] = (nn.Conv2d, nn.Linear),
    include_layers: list[str] | None = None,
) -> SensitivityReport:
    """Measure per-layer sensitivity of `model` under a lightweight perturbation.

    Args:
        model: The model to analyze. Not modified in place.
        eval_fn: Callable taking a model and returning a metrics dict.
            Must include `metric` as a key.
        metric: The metric to track (must exist in eval_fn's output).
        perturbation: "prune" removes the smallest-magnitude weights,
            "noise" adds Gaussian noise scaled by weight std.
        intensity: Perturbation strength in [0, 1]. For "prune" this is
            the sparsity fraction; for "noise" it scales the noise std.
        layer_types: Which module types to probe.
        include_layers: Optional allow-list of layer names. If None, all
            matching layer_types are analyzed.

    Returns:
        A SensitivityReport. Higher `metric_drop` means the layer is more
        sensitive to the chosen perturbation.
    """
    if perturbation not in _PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation {perturbation!r}; "
            f"expected one of {list(_PERTURBATIONS)}"
        )
    perturb_fn = _PERTURBATIONS[perturbation]

    baseline = eval_fn(model).get(metric)
    if baseline is None:
        raise KeyError(
            f"eval_fn did not return metric {metric!r}; "
            f"got keys: {list(eval_fn(model).keys())}"
        )

    targets: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, layer_types):
            continue
        if include_layers is not None and name not in include_layers:
            continue
        targets.append((name, module))

    layers: list[LayerSensitivity] = []
    for name, module in targets:
        # Snapshot & restore the single layer's weight to avoid deep-copying
        # the entire model once per layer.
        original = module.weight.data.clone()
        try:
            module.weight.data = perturb_fn(original, intensity)
            perturbed = eval_fn(model).get(metric, float("nan"))
        finally:
            module.weight.data = original

        layers.append(
            LayerSensitivity(
                name=name,
                layer_type=type(module).__name__,
                baseline_metric=baseline,
                perturbed_metric=perturbed,
                metric_drop=baseline - perturbed,
                num_params=sum(p.numel() for p in module.parameters()),
            )
        )

    return SensitivityReport(
        metric_name=metric,
        perturbation=perturbation,
        intensity=intensity,
        baseline_metric=baseline,
        layers=layers,
    )
