"""Tests for N:M sparsity stage."""

import pytest
import torch
import torch.nn as nn

from comprexx.stages.base import StageContext
from comprexx.stages.pruning.nm_sparsity import NMSparsity, _apply_nm_mask
from tests.fixtures.models import tiny_cnn


def _ctx():
    return StageContext(input_shape=(1, 3, 32, 32), device="cpu")


class TestNMMaskKernel:
    def test_2of4_exact(self):
        w = torch.tensor([[1.0, 2.0, 3.0, 4.0, 10.0, 9.0, 8.0, 7.0]])
        mask = _apply_nm_mask(w, n=2, m=4, random=False)
        # Group 1: {1,2,3,4} -> keep 3,4. Group 2: {10,9,8,7} -> keep 10,9.
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
        assert torch.equal(mask, expected)

    def test_mask_has_correct_density(self):
        w = torch.randn(8, 16)
        mask = _apply_nm_mask(w, n=2, m=4, random=False)
        # Every group of 4 along last dim should have exactly 2 ones
        groups = mask.reshape(8, -1, 4)
        assert torch.equal(groups.sum(-1), torch.full((8, 4), 2.0))


class TestNMSparsity:
    def test_basic_2of4(self):
        model = tiny_cnn()
        stage = NMSparsity(n=2, m=4)
        pruned, report = stage.apply(model, _ctx())

        assert report.stage_name == "nm_sparsity"
        assert report.technique == "nm_sparsity_2of4"
        # Verify pattern on an eligible conv/linear
        for m in pruned.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight.data.reshape(m.weight.shape[0], -1)
                if w.shape[1] >= 4 and w.shape[1] % 4 == 0:
                    groups = w.reshape(w.shape[0], -1, 4)
                    nonzero = (groups != 0).sum(-1)
                    assert (nonzero <= 2).all()

    def test_output_preserved(self):
        model = tiny_cnn()
        stage = NMSparsity(n=2, m=4)
        pruned, _ = stage.apply(model, _ctx())
        with torch.no_grad():
            out = pruned(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_1of4_higher_sparsity(self):
        model = tiny_cnn()
        stage = NMSparsity(n=1, m=4)
        pruned, report = stage.apply(model, _ctx())
        assert "1:4" in report.notes[-1]

    def test_invalid_n_m(self):
        with pytest.raises(ValueError):
            NMSparsity(n=4, m=4)

    def test_exclude_layers(self):
        model = tiny_cnn()
        original = model[0].weight.data.clone()
        stage = NMSparsity(n=2, m=4, exclude_layers=["0"])
        pruned, _ = stage.apply(model, _ctx())
        assert torch.equal(original, pruned[0].weight.data)

    def test_random_criteria(self):
        model = tiny_cnn()
        stage = NMSparsity(n=2, m=4, criteria="random")
        pruned, _ = stage.apply(model, _ctx())
        with torch.no_grad():
            out = pruned(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_original_model_unchanged(self):
        model = tiny_cnn()
        w_before = model[0].weight.data.clone()
        stage = NMSparsity(n=2, m=4)
        stage.apply(model, _ctx())
        assert torch.equal(model[0].weight.data, w_before)

    def test_recipe_integration(self, tmp_path):
        import comprexx as cx
        from comprexx.recipe.loader import recipe_to_pipeline

        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text(
            "name: t\n"
            "stages:\n"
            "  - technique: nm_sparsity\n"
            "    n: 2\n"
            "    m: 4\n"
        )
        recipe = cx.load_recipe(recipe_path)
        pipeline, _ = recipe_to_pipeline(recipe)
        result = pipeline.run(
            tiny_cnn(),
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path / "runs"),
        )
        assert len(result.report.stages) == 1
        assert result.report.stages[0].stage_name == "nm_sparsity"
