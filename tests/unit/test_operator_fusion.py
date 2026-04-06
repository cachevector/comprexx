"""Tests for operator fusion stage."""

import torch
import torch.nn as nn

from comprexx.stages.base import StageContext
from comprexx.stages.fusion.operator_fusion import (
    OperatorFusion,
    _fuse_conv_bn_eval,
)
from tests.fixtures.models import tiny_cnn


def _ctx():
    return StageContext(input_shape=(1, 3, 32, 32), device="cpu")


class TestConvBnFusionKernel:
    def test_fused_output_matches(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 8, 3, padding=1)
        bn = nn.BatchNorm2d(8)
        # Give BN some non-trivial running stats
        bn.running_mean = torch.randn(8)
        bn.running_var = torch.rand(8) + 0.5
        bn.weight.data = torch.randn(8)
        bn.bias.data = torch.randn(8)
        conv.eval()
        bn.eval()

        fused = _fuse_conv_bn_eval(conv, bn)
        fused.eval()

        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            ref = bn(conv(x))
            out = fused(x)
        assert torch.allclose(ref, out, atol=1e-5)


class TestOperatorFusion:
    def test_tiny_cnn_fusion(self):
        model = tiny_cnn().eval()
        n_bn_before = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        assert n_bn_before == 2

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            ref = model(x)

        stage = OperatorFusion()
        fused, report = stage.apply(model, _ctx())

        assert report.stage_name == "operator_fusion"
        assert "Fused 2" in " ".join(report.notes)

        n_bn_after = sum(1 for m in fused.modules() if isinstance(m, nn.BatchNorm2d))
        assert n_bn_after == 0

        with torch.no_grad():
            out = fused(x)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_param_count_reduces(self):
        model = tiny_cnn().eval()
        stage = OperatorFusion()
        _, report = stage.apply(model, _ctx())
        assert report.param_count_after < report.param_count_before

    def test_fallback_on_trace_error(self):
        """Non-traceable models should not crash the stage."""

        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.bn = nn.BatchNorm2d(8)

            def forward(self, x):
                # Data-dependent control flow breaks fx tracing
                if x.sum().item() > 0:
                    x = self.conv(x)
                else:
                    x = self.conv(x) * 2
                return self.bn(x)

        model = DynamicModel().eval()
        stage = OperatorFusion(fallback_on_trace_error=True)
        fused, report = stage.apply(model, _ctx())
        assert any("tracing failed" in n for n in report.notes)

    def test_original_model_unchanged(self):
        model = tiny_cnn().eval()
        n_before = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        OperatorFusion().apply(model, _ctx())
        n_after = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        assert n_after == n_before

    def test_recipe_integration(self, tmp_path):
        import comprexx as cx
        from comprexx.recipe.loader import recipe_to_pipeline

        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text(
            "name: t\n"
            "stages:\n"
            "  - technique: operator_fusion\n"
        )
        recipe = cx.load_recipe(recipe_path)
        pipeline, _ = recipe_to_pipeline(recipe)
        result = pipeline.run(
            tiny_cnn().eval(),
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path / "runs"),
        )
        assert len(result.report.stages) == 1
        assert result.report.stages[0].stage_name == "operator_fusion"
