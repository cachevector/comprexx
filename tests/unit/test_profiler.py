"""Tests for model profiler."""

import json

from comprexx.analysis.profiler import analyze


class TestAnalyze:
    def test_cnn_detection(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        assert profile.architecture_category == "cnn"

    def test_transformer_detection(self, transformer_model):
        profile = analyze(transformer_model, input_shape=(1, 8, 64))
        assert profile.architecture_category == "transformer"

    def test_rnn_detection(self, rnn_model):
        profile = analyze(rnn_model, input_shape=(1, 10, 32))
        assert profile.architecture_category == "rnn"

    def test_param_count(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        expected = sum(p.numel() for p in cnn_model.parameters())
        assert profile.total_params == expected

    def test_trainable_params(self, cnn_model):
        # Freeze one layer
        for p in list(cnn_model.parameters())[:1]:
            p.requires_grad = False
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        assert profile.trainable_params < profile.total_params

    def test_flops_positive(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        assert profile.total_flops > 0

    def test_size_bytes(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        assert profile.size_bytes > 0
        assert profile.size_mb > 0

    def test_compressible_layers(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        compressible = profile.compressible_layers()
        # Should include Conv2d and Linear, not BN/ReLU/Pool
        types = {l.layer_type for l in compressible}
        assert "Conv2d" in types
        assert "Linear" in types
        assert "ReLU" not in types
        assert "BatchNorm2d" not in types

    def test_custom_model_name(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32), model_name="my_cnn")
        assert profile.model_name == "my_cnn"

    def test_to_json(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        parsed = json.loads(profile.to_json())
        assert "total_params" in parsed
        assert "layers" in parsed
        assert len(parsed["layers"]) > 0

    def test_save(self, cnn_model, tmp_path):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        out = tmp_path / "profile.json"
        profile.save(out)
        parsed = json.loads(out.read_text())
        assert parsed["total_params"] == profile.total_params

    def test_summary(self, cnn_model):
        profile = analyze(cnn_model, input_shape=(1, 3, 32, 32))
        s = profile.summary()
        assert "cnn" in s
        assert "Parameters" in s
