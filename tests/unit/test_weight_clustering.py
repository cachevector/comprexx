"""Tests for weight clustering stage."""

import torch
import torch.nn as nn

from comprexx.stages.base import StageContext
from comprexx.stages.clustering.weight_clustering import (
    WeightClustering,
    _kmeans_1d,
)
from tests.fixtures.models import tiny_cnn


def _ctx():
    return StageContext(input_shape=(1, 3, 32, 32), device="cpu")


class TestKMeans1D:
    def test_unique_values_bounded(self):
        torch.manual_seed(0)
        w = torch.randn(64, 64)
        q = _kmeans_1d(w, k=8, init="linear", max_iter=10)
        assert q.shape == w.shape
        assert q.unique().numel() <= 8

    def test_single_value_handled(self):
        w = torch.full((8, 8), 3.14)
        q = _kmeans_1d(w, k=8, init="linear", max_iter=5)
        assert torch.all(q == 3.14)

    def test_approximates_input(self):
        torch.manual_seed(0)
        w = torch.randn(32, 32)
        q = _kmeans_1d(w, k=32, init="linear", max_iter=20)
        # With many centroids, error should be small
        assert (q - w).abs().mean().item() < 0.1


class TestWeightClustering:
    def test_basic_clustering(self):
        model = tiny_cnn()
        stage = WeightClustering(num_clusters=8)
        clustered, report = stage.apply(model, _ctx())

        assert report.stage_name == "weight_clustering"
        assert report.technique == "weight_clustering_k8"
        # Every clustered Conv/Linear weight has <= 8 unique values
        for m in clustered.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                assert m.weight.unique().numel() <= 8

    def test_size_reported_smaller(self):
        model = tiny_cnn()
        stage = WeightClustering(num_clusters=16)
        _, report = stage.apply(model, _ctx())
        assert report.size_bytes_after < report.size_bytes_before

    def test_output_shape_preserved(self):
        model = tiny_cnn()
        stage = WeightClustering(num_clusters=16)
        clustered, _ = stage.apply(model, _ctx())
        with torch.no_grad():
            out = clustered(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_output_close_with_many_clusters(self):
        torch.manual_seed(0)
        model = tiny_cnn().eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            ref = model(x)
        stage = WeightClustering(num_clusters=128)
        clustered, _ = stage.apply(model, _ctx())
        with torch.no_grad():
            out = clustered(x)
        assert (out - ref).abs().max().item() < 0.5

    def test_exclude_layers(self):
        model = tiny_cnn()
        original = model[0].weight.data.clone()
        stage = WeightClustering(num_clusters=4, exclude_layers=["0"])
        clustered, _ = stage.apply(model, _ctx())
        assert torch.equal(original, clustered[0].weight.data)

    def test_density_init(self):
        model = tiny_cnn()
        stage = WeightClustering(num_clusters=8, init="density")
        _, report = stage.apply(model, _ctx())
        assert report.stage_name == "weight_clustering"

    def test_original_model_unchanged(self):
        model = tiny_cnn()
        n_unique_before = model[0].weight.unique().numel()
        WeightClustering(num_clusters=4).apply(model, _ctx())
        assert model[0].weight.unique().numel() == n_unique_before

    def test_recipe_integration(self, tmp_path):
        import comprexx as cx
        from comprexx.recipe.loader import recipe_to_pipeline

        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text(
            "name: t\n"
            "stages:\n"
            "  - technique: weight_clustering\n"
            "    num_clusters: 8\n"
            "    init: linear\n"
        )
        recipe = cx.load_recipe(recipe_path)
        pipeline, _ = recipe_to_pipeline(recipe)
        result = pipeline.run(
            tiny_cnn(),
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path / "runs"),
        )
        assert len(result.report.stages) == 1
        assert result.report.stages[0].stage_name == "weight_clustering"
