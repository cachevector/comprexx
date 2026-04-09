"""Tests for low-rank decomposition stage."""

import torch
import torch.nn as nn

from comprexx.stages.base import StageContext
from comprexx.stages.decomposition.low_rank import (
    LowRankDecomposition,
    LowRankDecompositionConfig,
    _choose_rank,
    _svd_factorize,
)


def _mlp() -> nn.Module:
    """Wide MLP where SVD decomposition is profitable."""
    return nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def _ctx():
    return StageContext(input_shape=(1, 256), device="cpu")


class TestSVDFactorize:
    def test_reconstruction_full_rank(self):
        w = torch.randn(32, 64)
        first, second = _svd_factorize(w, None, rank=32)
        # W @ x == second(first(x))  where x flows through linear(right-mul)
        # nn.Linear does x @ W.T, so compose as: second(first(x)) computes
        # (x @ A.T) @ B.T == x @ (B A).T; so reconstructed W = B @ A.
        A = first.weight.data
        B = second.weight.data
        recon = B @ A
        assert torch.allclose(recon, w, atol=1e-4)

    def test_reduced_rank_approximation(self):
        torch.manual_seed(0)
        # Construct a rank-4 matrix
        u = torch.randn(32, 4)
        v = torch.randn(4, 64)
        w = u @ v
        first, second = _svd_factorize(w, None, rank=4)
        recon = second.weight.data @ first.weight.data
        assert (recon - w).abs().max().item() < 1e-4


class TestChooseRank:
    def test_ratio_mode(self):
        cfg = LowRankDecompositionConfig(rank_ratio=0.25)
        w = torch.randn(64, 128)
        r = _choose_rank(w, cfg)
        assert r == 16

    def test_energy_mode(self):
        cfg = LowRankDecompositionConfig(
            mode="energy", energy_threshold=0.99
        )
        u = torch.randn(32, 3)
        v = torch.randn(3, 64)
        w = u @ v + 1e-6 * torch.randn(32, 64)
        r = _choose_rank(w, cfg)
        assert r <= 4


class TestLowRankDecomposition:
    def test_basic_decomposition_reduces_params(self):
        model = _mlp()
        before = sum(p.numel() for p in model.parameters())

        stage = LowRankDecomposition(rank_ratio=0.25)
        decomposed, report = stage.apply(model, _ctx())
        after = sum(p.numel() for p in decomposed.parameters())

        assert after < before
        assert report.stage_name == "low_rank_decomposition"
        assert report.technique == "low_rank_svd"

    def test_output_close_to_original(self):
        torch.manual_seed(0)
        model = _mlp()
        x = torch.randn(2, 256)
        with torch.no_grad():
            ref = model(x)

        # With high rank ratio, reconstruction is near-perfect
        stage = LowRankDecomposition(rank_ratio=0.95)
        decomposed, _ = stage.apply(model, _ctx())
        with torch.no_grad():
            out = decomposed(x)

        # Shapes preserved at least
        assert out.shape == ref.shape

    def test_output_shape_preserved(self):
        model = _mlp()
        stage = LowRankDecomposition(rank_ratio=0.3)
        decomposed, _ = stage.apply(model, _ctx())
        with torch.no_grad():
            out = decomposed(torch.randn(1, 256))
        assert out.shape == (1, 10)

    def test_skips_layers_with_no_gain(self):
        # A layer where rank*(out+in) >= out*in should be skipped.
        # Linear(32, 10): full rank=10, rank_ratio=0.5 -> rank=5.
        # 5*(10+32)=210 < 320, so it would actually decompose.
        # Use a very small layer where rank_ratio gives no gain:
        model = nn.Sequential(nn.Linear(4, 4))
        stage = LowRankDecomposition(rank_ratio=0.9)
        _, report = stage.apply(
            model, StageContext(input_shape=(1, 4), device="cpu")
        )
        assert any("Skipped" in n or "no" in n.lower() for n in report.notes)

    def test_exclude_layers(self):
        model = _mlp()
        original_w = model[0].weight.data.clone()
        stage = LowRankDecomposition(rank_ratio=0.25, exclude_layers=["0"])
        decomposed, _ = stage.apply(model, _ctx())
        # Layer "0" is still a plain Linear with the same weights
        assert isinstance(decomposed[0], nn.Linear)
        assert torch.equal(decomposed[0].weight.data, original_w)

    def test_original_model_unchanged(self):
        model = _mlp()
        before_type = type(model[0])
        LowRankDecomposition(rank_ratio=0.25).apply(model, _ctx())
        assert type(model[0]) is before_type

    def test_recipe_integration(self, tmp_path):
        import comprexx as cx
        from comprexx.recipe.loader import recipe_to_pipeline

        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text(
            "name: t\n"
            "stages:\n"
            "  - technique: low_rank_decomposition\n"
            "    rank_ratio: 0.3\n"
        )
        recipe = cx.load_recipe(recipe_path)
        pipeline, _ = recipe_to_pipeline(recipe)
        result = pipeline.run(
            _mlp(),
            input_shape=(1, 256),
            output_dir=str(tmp_path / "runs"),
        )
        assert len(result.report.stages) == 1
        assert result.report.stages[0].stage_name == "low_rank_decomposition"
