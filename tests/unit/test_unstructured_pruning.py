"""Tests for unstructured pruning stage."""

import torch
import torch.nn as nn

from comprexx.stages.base import StageContext
from comprexx.stages.pruning.unstructured import UnstructuredPruning
from tests.fixtures.models import tiny_cnn


def _sparsity(model: nn.Module) -> float:
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            zeros += int((w == 0).sum().item())
            total += int(w.numel())
    return zeros / total if total else 0.0


class TestUnstructuredPruning:
    def _ctx(self):
        return StageContext(input_shape=(1, 3, 32, 32), device="cpu")

    def test_basic_magnitude_pruning(self):
        model = tiny_cnn()
        stage = UnstructuredPruning(sparsity=0.5, criteria="magnitude")
        pruned, report = stage.apply(model, self._ctx())

        assert report.stage_name == "unstructured_pruning"
        assert report.duration_seconds > 0
        # Effective sparsity should be near target
        assert _sparsity(pruned) >= 0.4

    def test_zero_sparsity(self):
        model = tiny_cnn()
        before = _sparsity(model)
        stage = UnstructuredPruning(sparsity=0.0)
        pruned, _ = stage.apply(model, self._ctx())
        assert abs(_sparsity(pruned) - before) < 1e-6

    def test_high_sparsity_still_runs(self):
        model = tiny_cnn()
        stage = UnstructuredPruning(sparsity=0.9)
        pruned, _ = stage.apply(model, self._ctx())
        with torch.no_grad():
            out = pruned(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)
        assert _sparsity(pruned) >= 0.8

    def test_local_scope(self):
        model = tiny_cnn()
        stage = UnstructuredPruning(sparsity=0.5, scope="local")
        pruned, _ = stage.apply(model, self._ctx())
        assert _sparsity(pruned) >= 0.4

    def test_gradual_pruning(self):
        model = tiny_cnn()
        stage = UnstructuredPruning(sparsity=0.6, gradual_steps=4)
        pruned, report = stage.apply(model, self._ctx())
        assert _sparsity(pruned) >= 0.5
        assert "4 step" in " ".join(report.notes)

    def test_exclude_layers(self):
        model = tiny_cnn()
        original = model[0].weight.data.clone()
        stage = UnstructuredPruning(sparsity=0.8, exclude_layers=["0"])
        pruned, _ = stage.apply(model, self._ctx())
        assert torch.equal(original, pruned[0].weight.data)

    def test_random_criteria(self):
        model = tiny_cnn()
        stage = UnstructuredPruning(sparsity=0.3, criteria="random")
        pruned, _ = stage.apply(model, self._ctx())
        assert _sparsity(pruned) >= 0.2

    def test_original_model_unchanged(self):
        model = tiny_cnn()
        before = _sparsity(model)
        stage = UnstructuredPruning(sparsity=0.7)
        stage.apply(model, self._ctx())
        assert _sparsity(model) == before

    def test_pruning_is_permanent(self):
        """No lingering _mask/_orig buffers after apply."""
        model = tiny_cnn()
        stage = UnstructuredPruning(sparsity=0.5)
        pruned, _ = stage.apply(model, self._ctx())
        for name, _ in pruned.named_buffers():
            assert "_mask" not in name

    def test_recipe_integration(self, tmp_path):
        import comprexx as cx
        from comprexx.recipe.loader import recipe_to_pipeline

        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text(
            "name: t\n"
            "stages:\n"
            "  - technique: unstructured_pruning\n"
            "    sparsity: 0.4\n"
            "    criteria: magnitude\n"
        )
        recipe = cx.load_recipe(recipe_path)
        pipeline, _ = recipe_to_pipeline(recipe)
        result = pipeline.run(
            tiny_cnn(),
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path / "runs"),
        )
        assert len(result.report.stages) == 1
        assert result.report.stages[0].stage_name == "unstructured_pruning"
