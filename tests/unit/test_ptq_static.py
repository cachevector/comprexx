"""Tests for PTQ static quantization."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from comprexx.core.exceptions import CalibrationError
from comprexx.stages.base import StageContext
from comprexx.stages.quantization.ptq_static import PTQStatic
from tests.fixtures.models import tiny_cnn


def _make_cal_loader(num_samples=32, batch_size=8):
    data = torch.randn(num_samples, 3, 32, 32)
    return DataLoader(TensorDataset(data), batch_size=batch_size)


class TestPTQStatic:
    def test_basic_quantization(self):
        model = tiny_cnn()
        stage = PTQStatic()
        ctx = StageContext(
            input_shape=(1, 3, 32, 32),
            device="cpu",
            calibration_data=_make_cal_loader(),
        )
        quantized, report = stage.apply(model, ctx)

        assert report.stage_name == "ptq_static"
        assert report.size_bytes_after <= report.size_bytes_before

    def test_missing_calibration_data(self):
        model = tiny_cnn()
        stage = PTQStatic()
        ctx = StageContext(input_shape=(1, 3, 32, 32), device="cpu")

        with pytest.raises(CalibrationError, match="calibration data"):
            stage.apply(model, ctx)

    def test_output_shape_preserved(self):
        model = tiny_cnn()
        stage = PTQStatic()
        ctx = StageContext(
            input_shape=(1, 3, 32, 32),
            device="cpu",
            calibration_data=_make_cal_loader(),
        )
        quantized, report = stage.apply(model, ctx)

        with torch.no_grad():
            out = quantized(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_report_notes(self):
        model = tiny_cnn()
        stage = PTQStatic(calibration_method="minmax")
        ctx = StageContext(
            input_shape=(1, 3, 32, 32),
            device="cpu",
            calibration_data=_make_cal_loader(),
        )
        _, report = stage.apply(model, ctx)

        assert any("minmax" in n for n in report.notes)
        assert any("Calibrated" in n for n in report.notes)

    def test_original_model_unchanged(self):
        model = tiny_cnn()
        original_params = sum(p.numel() for p in model.parameters())
        stage = PTQStatic()
        ctx = StageContext(
            input_shape=(1, 3, 32, 32),
            device="cpu",
            calibration_data=_make_cal_loader(),
        )
        stage.apply(model, ctx)

        assert sum(p.numel() for p in model.parameters()) == original_params
