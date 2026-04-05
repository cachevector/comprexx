"""Integration tests for the CLI."""

import json

import torch
from typer.testing import CliRunner

from comprexx.cli.main import app
from tests.fixtures.models import tiny_cnn

runner = CliRunner()


def _save_model(tmp_path) -> str:
    """Save a tiny model to a .pt file and return the path."""
    model = tiny_cnn()
    path = tmp_path / "model.pt"
    torch.save(model, path)
    return str(path)


class TestCLIAnalyze:
    def test_analyze_basic(self, tmp_path):
        model_path = _save_model(tmp_path)
        result = runner.invoke(
            app,
            ["analyze", model_path, "--input-shape", "1,3,32,32"],
        )
        assert result.exit_code == 0
        assert "Parameters" in result.output

    def test_analyze_json(self, tmp_path):
        model_path = _save_model(tmp_path)
        result = runner.invoke(
            app,
            ["analyze", model_path, "--input-shape", "1,3,32,32", "--json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "total_params" in parsed

    def test_analyze_verbose(self, tmp_path):
        model_path = _save_model(tmp_path)
        result = runner.invoke(
            app,
            ["analyze", model_path, "--input-shape", "1,3,32,32", "--verbose"],
        )
        assert result.exit_code == 0
        assert "Layer" in result.output

    def test_analyze_save(self, tmp_path):
        model_path = _save_model(tmp_path)
        out = tmp_path / "profile.json"
        result = runner.invoke(
            app,
            ["analyze", model_path, "--input-shape", "1,3,32,32", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()


class TestCLICompress:
    def test_compress_with_recipe(self, tmp_path):
        model_path = _save_model(tmp_path)
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(
            "name: test\n"
            "stages:\n"
            "  - technique: structured_pruning\n"
            "    sparsity: 0.2\n"
        )
        result = runner.invoke(
            app,
            [
                "compress",
                model_path,
                "--recipe",
                str(recipe),
                "--input-shape",
                "1,3,32,32",
                "--output-dir",
                str(tmp_path / "runs"),
            ],
        )
        assert result.exit_code == 0

    def test_compress_dry_run(self, tmp_path):
        model_path = _save_model(tmp_path)
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(
            "name: test\n"
            "stages:\n"
            "  - technique: structured_pruning\n"
            "    sparsity: 0.3\n"
        )
        result = runner.invoke(
            app,
            [
                "compress",
                model_path,
                "--recipe",
                str(recipe),
                "--input-shape",
                "1,3,32,32",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "runs"),
            ],
        )
        assert result.exit_code == 0


class TestCLIExport:
    def test_export_unsupported_format(self, tmp_path):
        model_path = _save_model(tmp_path)
        result = runner.invoke(
            app,
            ["export", model_path, "--format", "tflite", "--input-shape", "1,3,32,32"],
        )
        assert result.exit_code == 1
