"""Tests for the benchmark module."""

import pytest
import torch

import comprexx as cx
from comprexx.benchmark.runner import BenchmarkComparison, BenchmarkResult, _percentile
from tests.fixtures.models import tiny_cnn, tiny_transformer


class TestBenchmark:
    def test_basic_run(self):
        model = tiny_cnn()
        result = cx.benchmark(model, input_shape=(1, 3, 32, 32), warmup=2, iters=5)
        assert isinstance(result, BenchmarkResult)
        assert result.iters == 5
        assert result.warmup == 2
        assert result.batch_size == 1
        assert result.mean_ms > 0
        assert result.throughput_ips > 0
        assert len(result.samples_ms) == 5

    def test_batch_size_from_shape(self):
        model = tiny_cnn()
        result = cx.benchmark(model, input_shape=(4, 3, 32, 32), warmup=1, iters=3)
        assert result.batch_size == 4

    def test_percentiles_ordered(self):
        model = tiny_cnn()
        result = cx.benchmark(model, input_shape=(1, 3, 32, 32), warmup=1, iters=10)
        assert result.p50_ms <= result.p90_ms <= result.p99_ms
        assert result.min_ms <= result.mean_ms <= result.max_ms

    def test_invalid_iters(self):
        model = tiny_cnn()
        with pytest.raises(ValueError):
            cx.benchmark(model, input_shape=(1, 3, 32, 32), iters=0)

    def test_transformer_multi_input_shape(self):
        model = tiny_transformer()
        # (batch, seq, features)
        result = cx.benchmark(model, input_shape=(2, 8, 64), warmup=1, iters=3)
        assert result.batch_size == 2
        assert result.mean_ms > 0

    def test_custom_input_fn(self):
        model = tiny_cnn()

        def fn():
            return torch.randn(1, 3, 32, 32)

        result = cx.benchmark(model, input_shape=(1, 3, 32, 32), input_fn=fn,
                              warmup=1, iters=3)
        assert result.mean_ms > 0

    def test_summary_and_serialization(self):
        model = tiny_cnn()
        result = cx.benchmark(model, input_shape=(1, 3, 32, 32), warmup=1, iters=3)
        s = result.summary()
        assert "Mean" in s and "ms" in s
        d = result.to_dict()
        assert d["iters"] == 3
        j = result.to_json()
        assert "mean_ms" in j


class TestCompare:
    def test_compare_same_model(self):
        m = tiny_cnn()
        cmp = cx.compare_benchmarks(m, m, input_shape=(1, 3, 32, 32),
                                    warmup=1, iters=5)
        assert isinstance(cmp, BenchmarkComparison)
        # Same model: speedup should be in a sane range
        assert 0.2 < cmp.speedup < 5.0

    def test_compare_summary(self):
        m = tiny_cnn()
        cmp = cx.compare_benchmarks(m, m, input_shape=(1, 3, 32, 32),
                                    warmup=1, iters=3)
        s = cmp.summary()
        assert "Baseline" in s and "Compressed" in s
        assert "speedup" in cmp.to_dict()


class TestPercentile:
    def test_basic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0
        assert _percentile(vals, 0) == 1.0
        assert _percentile(vals, 100) == 5.0

    def test_empty(self):
        assert _percentile([], 50) == 0.0
