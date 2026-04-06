"""Tests for per-layer sensitivity analysis."""

import math

import pytest
import torch
import torch.nn as nn

from comprexx.analysis.sensitivity import (
    LayerSensitivity,
    SensitivityReport,
    _perturb_noise,
    _perturb_prune,
    analyze_sensitivity,
)
from tests.fixtures.models import tiny_cnn


def _make_eval_fn(ref_output: torch.Tensor, x: torch.Tensor):
    """Build an eval_fn that measures output drift as a proxy metric.

    Uses exp(-mse) so that the baseline is ~1 and any drift drives it
    toward 0. This lets sensitivity tests run without a real dataset.
    """

    def eval_fn(model: nn.Module) -> dict[str, float]:
        with torch.no_grad():
            out = model(x)
        mse = (out - ref_output).pow(2).mean().item()
        return {"drift_score": math.exp(-mse)}

    return eval_fn


class TestPerturbations:
    def test_prune_zeros_fraction(self):
        w = torch.randn(100)
        p = _perturb_prune(w, 0.3)
        n_zero = (p == 0).sum().item()
        assert 25 <= n_zero <= 35

    def test_noise_changes_weights(self):
        torch.manual_seed(0)
        w = torch.randn(64, 64)
        p = _perturb_noise(w, 0.1)
        assert not torch.equal(p, w)
        # Noise should be roughly proportional to std * intensity
        delta = (p - w).abs().mean().item()
        assert 0.0 < delta < 1.0


class TestSensitivityReport:
    def _fake(self, drops: list[float]) -> SensitivityReport:
        return SensitivityReport(
            metric_name="m",
            perturbation="prune",
            intensity=0.3,
            baseline_metric=1.0,
            layers=[
                LayerSensitivity(
                    name=f"layer_{i}",
                    layer_type="Linear",
                    baseline_metric=1.0,
                    perturbed_metric=1.0 - d,
                    metric_drop=d,
                    num_params=100,
                )
                for i, d in enumerate(drops)
            ],
        )

    def test_most_sensitive_ordering(self):
        r = self._fake([0.1, 0.5, 0.2, 0.4])
        top = r.most_sensitive(2)
        assert [l.metric_drop for l in top] == [0.5, 0.4]

    def test_most_tolerant_ordering(self):
        r = self._fake([0.1, 0.5, 0.2, 0.4])
        bottom = r.most_tolerant(2)
        assert [l.metric_drop for l in bottom] == [0.1, 0.2]

    def test_recommend_exclusions(self):
        r = self._fake([0.01, 0.3, 0.05, 0.4])
        excl = r.recommend_exclusions(threshold=0.1)
        assert set(excl) == {"layer_1", "layer_3"}

    def test_to_dict_roundtrip(self):
        r = self._fake([0.1, 0.2])
        d = r.to_dict()
        assert d["baseline_metric"] == 1.0
        assert len(d["layers"]) == 2
        assert d["layers"][0]["metric_drop"] == 0.1

    def test_summary_runs(self):
        r = self._fake([0.1, 0.2, 0.3])
        s = r.summary()
        assert "most sensitive" in s
        assert "layer_2" in s


class TestAnalyzeSensitivity:
    def test_basic_prune_on_tiny_cnn(self):
        torch.manual_seed(0)
        model = tiny_cnn().eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            ref = model(x)

        report = analyze_sensitivity(
            model,
            eval_fn=_make_eval_fn(ref, x),
            metric="drift_score",
            perturbation="prune",
            intensity=0.5,
        )

        # One entry per Conv2d + Linear (tiny_cnn has 2 convs + 1 linear)
        assert len(report.layers) == 3
        # Baseline should be ~1.0 (model unchanged)
        assert abs(report.baseline_metric - 1.0) < 1e-6
        # At least one layer should have measurable drop
        assert max(l.metric_drop for l in report.layers) > 0
        # Model weights should be restored after analysis
        with torch.no_grad():
            after = model(x)
        assert torch.allclose(after, ref)

    def test_noise_perturbation(self):
        torch.manual_seed(0)
        model = tiny_cnn().eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            ref = model(x)

        report = analyze_sensitivity(
            model,
            eval_fn=_make_eval_fn(ref, x),
            metric="drift_score",
            perturbation="noise",
            intensity=0.5,
        )
        assert report.perturbation == "noise"
        assert len(report.layers) == 3

    def test_include_layers_filter(self):
        torch.manual_seed(0)
        model = tiny_cnn().eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            ref = model(x)

        report = analyze_sensitivity(
            model,
            eval_fn=_make_eval_fn(ref, x),
            metric="drift_score",
            include_layers=["0"],
        )
        assert len(report.layers) == 1
        assert report.layers[0].name == "0"

    def test_invalid_perturbation(self):
        with pytest.raises(ValueError):
            analyze_sensitivity(
                tiny_cnn(),
                eval_fn=lambda m: {"drift_score": 1.0},
                metric="drift_score",
                perturbation="bogus",  # type: ignore[arg-type]
            )

    def test_missing_metric_raises(self):
        with pytest.raises(KeyError):
            analyze_sensitivity(
                tiny_cnn(),
                eval_fn=lambda m: {"other": 1.0},
                metric="drift_score",
            )

    def test_public_api_export(self):
        import comprexx as cx

        assert hasattr(cx, "analyze_sensitivity")
        assert hasattr(cx, "SensitivityReport")
