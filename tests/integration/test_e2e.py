"""End-to-end integration test — the golden path."""

from pathlib import Path

import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")


class TestEndToEnd:
    def test_analyze_compress_export(self, tmp_path):
        """Full v0.1 golden path: analyze -> pipeline -> export."""
        import comprexx as cx
        from tests.fixtures.models import tiny_cnn

        # 1. Create and analyze model
        model = tiny_cnn()
        profile = cx.analyze(model, input_shape=(1, 3, 32, 32))

        assert profile.total_params > 0
        assert profile.architecture_category == "cnn"
        assert len(profile.compressible_layers()) > 0

        # 2. Build and run pipeline
        pipeline = cx.Pipeline([
            cx.stages.StructuredPruning(sparsity=0.3, criteria="l1_norm"),
        ])

        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path / "runs"),
        )

        assert len(result.report.stages) == 1
        assert result.report.total_duration_seconds > 0

        # Compressed model still works
        with torch.no_grad():
            out = result.model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

        # 3. Export to ONNX
        onnx_path = tmp_path / "model.onnx"
        exporter = cx.ONNXExporter(opset_version=14, simplify=False)
        manifest = exporter.export(
            result.model,
            input_shape=(1, 3, 32, 32),
            output_path=str(onnx_path),
            compression_report=result.report,
        )

        assert onnx_path.exists()
        assert manifest.export_format == "onnx"
        assert "compression_ratio" in manifest.compression_stats

        # 4. Verify ONNX model works
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        dummy = torch.randn(1, 3, 32, 32).numpy()
        ort_out = session.run(None, {"input": dummy})[0]
        assert ort_out.shape == (1, 10)

    def test_recipe_driven_pipeline(self, tmp_path):
        """Test loading a recipe and running the pipeline."""
        import comprexx as cx
        from tests.fixtures.models import tiny_cnn

        recipe_path = tmp_path / "recipe.yaml"
        recipe_path.write_text(
            "name: test-e2e\n"
            "stages:\n"
            "  - technique: structured_pruning\n"
            "    sparsity: 0.2\n"
            "  - technique: ptq_dynamic\n"
            "    format: int8\n"
        )

        recipe = cx.load_recipe(recipe_path)
        from comprexx.recipe.loader import recipe_to_pipeline

        pipeline, guard = recipe_to_pipeline(recipe)

        model = tiny_cnn()
        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path / "runs"),
        )

        assert len(result.report.stages) == 2
        assert Path(result.run_dir).exists()
