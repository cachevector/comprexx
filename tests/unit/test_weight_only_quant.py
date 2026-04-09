"""Tests for weight-only quantization stage."""

import torch

from comprexx.stages.base import StageContext
from comprexx.stages.quantization.weight_only import (
    WeightOnlyQuant,
    _quantize_tensor,
)
from tests.fixtures.models import tiny_cnn


def _ctx():
    return StageContext(input_shape=(1, 3, 32, 32), device="cpu")


class TestQuantizeTensor:
    def test_int8_symmetric_roundtrip_close(self):
        torch.manual_seed(0)
        w = torch.randn(8, 64)
        dq, scale, zero = _quantize_tensor(w, bits=8, group_size=32, symmetric=True)
        assert zero is None
        assert dq.shape == w.shape
        err = (dq - w).abs().max().item()
        # INT8 symmetric within a group of 32 should have small max error
        assert err < 0.05

    def test_int4_has_few_unique_values(self):
        w = torch.randn(4, 32)
        dq, scale, _ = _quantize_tensor(w, bits=4, group_size=32, symmetric=True)
        # With 1 group per row, at most 2^4=16 levels per row
        for row in dq:
            assert row.unique().numel() <= 16

    def test_asymmetric_supported(self):
        w = torch.randn(4, 32) + 5.0  # shifted
        dq, scale, zero = _quantize_tensor(w, bits=4, group_size=32, symmetric=False)
        assert zero is not None
        assert (dq - w).abs().max().item() < 1.0

    def test_padding_nondivisible(self):
        w = torch.randn(2, 30)  # 30 not divisible by 32
        dq, _, _ = _quantize_tensor(w, bits=8, group_size=32, symmetric=True)
        assert dq.shape == w.shape


class TestWeightOnlyQuant:
    def test_basic_int8(self):
        model = tiny_cnn()
        stage = WeightOnlyQuant(bits=8, group_size=64)
        quantized, report = stage.apply(model, _ctx())

        assert report.stage_name == "weight_only_quant"
        assert report.technique == "weight_only_int8"
        assert report.size_bytes_after < report.size_bytes_before
        # Model still runs
        with torch.no_grad():
            out = quantized(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_int4_smaller_than_int8(self):
        m1 = tiny_cnn()
        m2 = tiny_cnn()
        # Use same weights for a fair comparison
        m2.load_state_dict(m1.state_dict())

        int8, r8 = WeightOnlyQuant(bits=8, group_size=64).apply(m1, _ctx())
        int4, r4 = WeightOnlyQuant(bits=4, group_size=64).apply(m2, _ctx())

        assert r4.size_bytes_after < r8.size_bytes_after

    def test_output_close_to_original_int8(self):
        model = tiny_cnn()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            ref = model(x)
        stage = WeightOnlyQuant(bits=8, group_size=64)
        quantized, _ = stage.apply(model, _ctx())
        with torch.no_grad():
            q_out = quantized(x)
        # INT8 should be quite close to fp32
        assert (q_out - ref).abs().max().item() < 0.5

    def test_exclude_layers(self):
        model = tiny_cnn()
        original = model[0].weight.data.clone()
        stage = WeightOnlyQuant(bits=4, exclude_layers=["0"])
        pruned, _ = stage.apply(model, _ctx())
        assert torch.equal(original, pruned[0].weight.data)

    def test_original_model_unchanged(self):
        model = tiny_cnn()
        w_before = model[0].weight.data.clone()
        WeightOnlyQuant(bits=4).apply(model, _ctx())
        assert torch.equal(model[0].weight.data, w_before)

    def test_report_size_savings(self):
        model = tiny_cnn()
        stage = WeightOnlyQuant(bits=4, group_size=64)
        _, report = stage.apply(model, _ctx())
        # ~4x compression on the quantized layers (minus scale overhead)
        assert report.compression_ratio > 1.5

    def test_recipe_integration(self, tmp_path):
        import comprexx as cx
        from comprexx.recipe.loader import recipe_to_pipeline

        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text(
            "name: t\n"
            "stages:\n"
            "  - technique: weight_only_quant\n"
            "    bits: 4\n"
            "    group_size: 64\n"
        )
        recipe = cx.load_recipe(recipe_path)
        pipeline, _ = recipe_to_pipeline(recipe)
        result = pipeline.run(
            tiny_cnn(),
            input_shape=(1, 3, 32, 32),
            output_dir=str(tmp_path / "runs"),
        )
        assert len(result.report.stages) == 1
        assert result.report.stages[0].technique == "weight_only_int4"
