"""Tests for structured pruning stage."""

import torch

from comprexx.stages.base import StageContext
from comprexx.stages.pruning.structured import StructuredPruning
from tests.fixtures.models import tiny_cnn


class TestStructuredPruning:
    def _make_context(self, input_shape=(1, 3, 32, 32)):
        return StageContext(input_shape=input_shape, device="cpu")

    def test_basic_pruning(self):
        model = tiny_cnn()
        stage = StructuredPruning(sparsity=0.5, criteria="l1_norm")
        ctx = self._make_context()
        pruned, report = stage.apply(model, ctx)

        assert report.size_bytes_after <= report.size_bytes_before
        assert report.stage_name == "structured_pruning"
        assert report.duration_seconds > 0

    def test_zero_sparsity(self):
        model = tiny_cnn()
        stage = StructuredPruning(sparsity=0.0)
        ctx = self._make_context()
        pruned, report = stage.apply(model, ctx)

        # No pruning applied
        assert report.size_bytes_after == report.size_bytes_before

    def test_high_sparsity(self):
        model = tiny_cnn()
        stage = StructuredPruning(sparsity=0.9, criteria="l1_norm")
        ctx = self._make_context()
        pruned, report = stage.apply(model, ctx)

        # Should still produce a valid model
        with torch.no_grad():
            out = pruned(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_exclude_layers(self):
        model = tiny_cnn()
        original_first_conv_weight = model[0].weight.data.clone()

        # Exclude the first conv layer ("0" in Sequential)
        stage = StructuredPruning(sparsity=0.5, exclude_layers=["0"])
        ctx = self._make_context()
        pruned, report = stage.apply(model, ctx)

        # Verify first conv layer is unchanged
        pruned_first_conv_weight = pruned[0].weight.data
        assert torch.equal(original_first_conv_weight, pruned_first_conv_weight)

    def test_l2_norm_criteria(self):
        model = tiny_cnn()
        stage = StructuredPruning(sparsity=0.3, criteria="l2_norm")
        ctx = self._make_context()
        pruned, report = stage.apply(model, ctx)
        assert report.stage_name == "structured_pruning"

    def test_local_scope(self):
        model = tiny_cnn()
        stage = StructuredPruning(sparsity=0.3, scope="local")
        ctx = self._make_context()
        pruned, report = stage.apply(model, ctx)
        assert "Local pruning" in report.notes[0]

    def test_global_scope(self):
        model = tiny_cnn()
        stage = StructuredPruning(sparsity=0.3, scope="global")
        ctx = self._make_context()
        pruned, report = stage.apply(model, ctx)
        assert "Global pruning" in report.notes[0]

    def test_report_fields(self):
        model = tiny_cnn()
        stage = StructuredPruning(sparsity=0.3)
        ctx = self._make_context()
        _, report = stage.apply(model, ctx)

        assert report.param_count_before > 0
        assert report.flops_before > 0
        assert report.size_bytes_before > 0
        assert len(report.notes) > 0

    def test_original_model_unchanged(self):
        model = tiny_cnn()
        original_params = sum(p.numel() for p in model.parameters())
        stage = StructuredPruning(sparsity=0.5)
        ctx = self._make_context()
        stage.apply(model, ctx)

        # Original model should not be modified (deep copy inside apply)
        assert sum(p.numel() for p in model.parameters()) == original_params
