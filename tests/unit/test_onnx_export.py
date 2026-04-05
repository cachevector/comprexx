"""Tests for ONNX export."""

import json
from pathlib import Path

import pytest
import torch

from comprexx.core.exceptions import ExportError
from comprexx.export.onnx import ONNXExporter
from tests.fixtures.models import tiny_cnn

onnxruntime = pytest.importorskip("onnxruntime")


class TestONNXExporter:
    def test_basic_export(self, tmp_path):
        model = tiny_cnn()
        exporter = ONNXExporter(opset_version=14, simplify=False)
        out = tmp_path / "model.onnx"
        manifest = exporter.export(model, input_shape=(1, 3, 32, 32), output_path=str(out))

        assert out.exists()
        assert (tmp_path / "comprexx_manifest.json").exists()
        assert manifest.export_format == "onnx"

    def test_manifest_content(self, tmp_path):
        model = tiny_cnn()
        exporter = ONNXExporter(opset_version=14, simplify=False)
        out = tmp_path / "model.onnx"
        exporter.export(model, input_shape=(1, 3, 32, 32), output_path=str(out))

        manifest = json.loads((tmp_path / "comprexx_manifest.json").read_text())
        assert "original_model_name" in manifest
        assert "original_model_hash" in manifest
        assert "comprexx_version" in manifest
        assert "timestamp" in manifest

    def test_output_matches_pytorch(self, tmp_path):
        model = tiny_cnn()
        model.eval()
        exporter = ONNXExporter(opset_version=14, simplify=False, validate=False)
        out = tmp_path / "model.onnx"
        exporter.export(model, input_shape=(1, 3, 32, 32), output_path=str(out))

        # Compare outputs
        import onnxruntime as ort

        dummy = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            pt_out = model(dummy).numpy()

        session = ort.InferenceSession(str(out))
        ort_out = session.run(None, {"input": dummy.numpy()})[0]

        max_diff = abs(pt_out - ort_out).max()
        assert max_diff < 1e-4

    def test_creates_parent_dirs(self, tmp_path):
        model = tiny_cnn()
        exporter = ONNXExporter(opset_version=14, simplify=False, validate=False)
        out = tmp_path / "subdir" / "deep" / "model.onnx"
        exporter.export(model, input_shape=(1, 3, 32, 32), output_path=str(out))
        assert out.exists()

    def test_export_error_on_bad_model(self, tmp_path):
        class BadModel(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("intentional failure")

        model = BadModel()
        exporter = ONNXExporter(opset_version=14, simplify=False, validate=False)
        out = tmp_path / "model.onnx"

        with pytest.raises(ExportError):
            exporter.export(model, input_shape=(1, 3), output_path=str(out))
