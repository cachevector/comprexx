"""Tests for PTQ dynamic quantization."""

import torch

from comprexx.stages.base import StageContext
from comprexx.stages.quantization.ptq_dynamic import PTQDynamic
from tests.fixtures.models import tiny_cnn


class TestPTQDynamic:
    def _make_context(self, input_shape=(1, 3, 32, 32)):
        return StageContext(input_shape=input_shape, device="cpu")

    def test_basic_quantization(self):
        model = tiny_cnn()
        stage = PTQDynamic()
        ctx = self._make_context()
        quantized, report = stage.apply(model, ctx)

        assert report.stage_name == "ptq_dynamic"
        assert report.size_bytes_after <= report.size_bytes_before

    def test_output_shape_preserved(self):
        model = tiny_cnn()
        stage = PTQDynamic()
        ctx = self._make_context()
        quantized, report = stage.apply(model, ctx)

        with torch.no_grad():
            out = quantized(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_report_fields(self):
        model = tiny_cnn()
        stage = PTQDynamic()
        ctx = self._make_context()
        _, report = stage.apply(model, ctx)

        assert report.duration_seconds > 0
        assert report.param_count_before > 0
        assert report.size_bytes_before > 0
        assert len(report.notes) > 0

    def test_original_model_unchanged(self):
        model = tiny_cnn()
        original_type = type(list(model.modules())[1])  # First actual layer
        stage = PTQDynamic()
        ctx = self._make_context()
        stage.apply(model, ctx)

        # Original model layers should not be replaced
        assert type(list(model.modules())[1]) == original_type
