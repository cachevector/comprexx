"""Integration tests for the Pipeline engine."""

import json
from pathlib import Path

import pytest
import torch

from comprexx.core.exceptions import AccuracyGuardTriggered
from comprexx.core.guard import AccuracyGuard
from comprexx.core.pipeline import Pipeline
from comprexx.stages.pruning.structured import StructuredPruning
from comprexx.stages.quantization.ptq_dynamic import PTQDynamic
from tests.fixtures.models import tiny_cnn


class TestPipeline:
    def test_single_stage(self, tmp_path):
        model = tiny_cnn()
        pipeline = Pipeline([StructuredPruning(sparsity=0.3)])
        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path),
        )

        assert len(result.report.stages) == 1
        assert result.report.stages[0].stage_name == "structured_pruning"
        assert Path(result.run_dir).exists()

    def test_multi_stage(self, tmp_path):
        model = tiny_cnn()
        pipeline = Pipeline([
            StructuredPruning(sparsity=0.3),
            PTQDynamic(),
        ])
        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path),
        )

        assert len(result.report.stages) == 2
        # Model should still produce valid output
        with torch.no_grad():
            out = result.model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_dry_run(self, tmp_path):
        model = tiny_cnn()
        original_params = sum(p.numel() for p in model.parameters())

        pipeline = Pipeline([StructuredPruning(sparsity=0.5)])
        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            dry_run=True,
            output_dir=str(tmp_path),
        )

        # Model should be unchanged in dry run
        assert sum(p.numel() for p in result.model.parameters()) == original_params
        assert len(result.report.stages) == 1

    def test_accuracy_guard_halts(self, tmp_path):
        model = tiny_cnn()

        call_count = 0

        def eval_fn(m):
            nonlocal call_count
            call_count += 1
            # Simulate accuracy degradation
            if call_count == 1:
                return {"top1_accuracy": 0.95}  # baseline
            return {"top1_accuracy": 0.80}  # after compression

        pipeline = Pipeline([
            StructuredPruning(sparsity=0.5),
            PTQDynamic(),  # should not be reached
        ])

        with pytest.raises(AccuracyGuardTriggered) as exc_info:
            pipeline.run(
                model,
                input_shape=(1, 3, 32, 32),
                eval_fn=eval_fn,
                accuracy_guard=AccuracyGuard(max_drop=0.01, action="halt"),
                output_dir=str(tmp_path),
            )

        assert exc_info.value.stage == "structured_pruning"

    def test_run_directory_structure(self, tmp_path):
        model = tiny_cnn()
        pipeline = Pipeline([StructuredPruning(sparsity=0.3)])
        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path),
        )

        run_dir = Path(result.run_dir)
        assert (run_dir / "model_profile.json").exists()
        assert (run_dir / "compression_report.json").exists()
        assert (run_dir / "stage_reports").is_dir()

        # Verify JSON is valid
        profile = json.loads((run_dir / "model_profile.json").read_text())
        assert "total_params" in profile

        report = json.loads((run_dir / "compression_report.json").read_text())
        assert report["model_name"] == "Sequential"

    def test_compression_report_metrics(self, tmp_path):
        model = tiny_cnn()
        pipeline = Pipeline([StructuredPruning(sparsity=0.3)])
        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path),
        )

        assert result.report.total_duration_seconds > 0
        assert result.profile_before.total_params > 0

    def test_empty_pipeline(self, tmp_path):
        model = tiny_cnn()
        pipeline = Pipeline([])
        result = pipeline.run(
            model,
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path),
        )

        assert len(result.report.stages) == 0
        assert result.report.total_compression_ratio == 1.0
