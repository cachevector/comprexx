"""Tests for FLOPs counter."""

import torch.nn as nn

from comprexx.analysis.flops import count_flops


class TestCountFlops:
    def test_linear_flops(self):
        model = nn.Linear(100, 50)
        total, per_layer = count_flops(model, input_shape=(1, 100))
        # 2 * 1 * 100 * 50 = 10000, plus bias = 10050
        assert total > 0
        assert len(per_layer) == 1

    def test_conv2d_flops(self):
        model = nn.Conv2d(3, 16, 3, padding=1)
        total, per_layer = count_flops(model, input_shape=(1, 3, 32, 32))
        assert total > 0
        assert len(per_layer) == 1

    def test_sequential_model(self, cnn_model):
        total, per_layer = count_flops(cnn_model, input_shape=(1, 3, 32, 32))
        assert total > 0
        # Should have entries for Conv2d, Linear, and BN layers
        assert len(per_layer) >= 3

    def test_per_layer_sum_equals_total(self, cnn_model):
        total, per_layer = count_flops(cnn_model, input_shape=(1, 3, 32, 32))
        assert total == sum(per_layer.values())

    def test_no_hooks_leak(self, cnn_model):
        """Hooks should be cleaned up after counting."""
        hook_count_before = sum(
            len(m._forward_hooks) for m in cnn_model.modules()
        )
        count_flops(cnn_model, input_shape=(1, 3, 32, 32))
        hook_count_after = sum(
            len(m._forward_hooks) for m in cnn_model.modules()
        )
        assert hook_count_before == hook_count_after
