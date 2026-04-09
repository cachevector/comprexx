"""Latency/throughput benchmarking for PyTorch models.

Measures real inference performance so compression reports can show actual
speedups, not just parameter-count reductions.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Latency statistics from a benchmark run."""

    device: str
    dtype: str
    batch_size: int
    warmup: int
    iters: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    throughput_ips: float
    samples_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        return (
            f"Benchmark ({self.device}, batch={self.batch_size}, iters={self.iters})\n"
            f"  Mean:       {self.mean_ms:.3f} ms\n"
            f"  Median:     {self.median_ms:.3f} ms\n"
            f"  Std:        {self.std_ms:.3f} ms\n"
            f"  p50/p90/p99:{self.p50_ms:.3f} / {self.p90_ms:.3f} / {self.p99_ms:.3f} ms\n"
            f"  Min/Max:    {self.min_ms:.3f} / {self.max_ms:.3f} ms\n"
            f"  Throughput: {self.throughput_ips:.1f} inferences/sec"
        )


@dataclass
class BenchmarkComparison:
    """Before/after benchmark comparison."""

    baseline: BenchmarkResult
    compressed: BenchmarkResult

    @property
    def speedup(self) -> float:
        if self.compressed.mean_ms == 0:
            return float("inf")
        return self.baseline.mean_ms / self.compressed.mean_ms

    @property
    def latency_reduction_pct(self) -> float:
        if self.baseline.mean_ms == 0:
            return 0.0
        return (1 - self.compressed.mean_ms / self.baseline.mean_ms) * 100

    @property
    def throughput_gain_pct(self) -> float:
        if self.baseline.throughput_ips == 0:
            return 0.0
        return (self.compressed.throughput_ips / self.baseline.throughput_ips - 1) * 100

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "compressed": self.compressed.to_dict(),
            "speedup": self.speedup,
            "latency_reduction_pct": self.latency_reduction_pct,
            "throughput_gain_pct": self.throughput_gain_pct,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        return (
            f"Benchmark Comparison ({self.baseline.device})\n"
            f"  Baseline:   {self.baseline.mean_ms:.3f} ms  "
            f"({self.baseline.throughput_ips:.1f} ips)\n"
            f"  Compressed: {self.compressed.mean_ms:.3f} ms  "
            f"({self.compressed.throughput_ips:.1f} ips)\n"
            f"  Speedup:    {self.speedup:.2f}x  "
            f"({self.latency_reduction_pct:+.1f}% latency, "
            f"{self.throughput_gain_pct:+.1f}% throughput)"
        )


def _make_input(
    input_shape: tuple[int, ...] | list[tuple[int, ...]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(input_shape, list):
        return tuple(torch.randn(*s, device=device, dtype=dtype) for s in input_shape)
    return torch.randn(*input_shape, device=device, dtype=dtype)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def benchmark(
    model: nn.Module,
    input_shape: tuple[int, ...] | list[tuple[int, ...]],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    iters: int = 50,
    input_fn: Optional[Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]] = None,
) -> BenchmarkResult:
    """Benchmark a model's inference latency.

    Args:
        model: Model to benchmark (set to eval mode internally).
        input_shape: Single tensor shape, or list of shapes for multi-input models.
        device: "cpu" or "cuda".
        dtype: Input tensor dtype. Quantized models ignore this.
        warmup: Warmup iterations (not measured) to stabilize caches/JIT.
        iters: Measured iterations.
        input_fn: Optional callable returning a fresh input per call. Overrides
            `input_shape` when provided.
    """
    if iters <= 0:
        raise ValueError("iters must be positive")

    dev = torch.device(device)
    model = model.eval()

    # Quantized models must run on CPU
    is_quantized = any(
        "quantized" in type(m).__module__ for m in model.modules()
    )
    if is_quantized and dev.type != "cpu":
        dev = torch.device("cpu")

    try:
        model = model.to(dev)
    except (RuntimeError, NotImplementedError):
        # Some quantized modules refuse .to() transfers; fall through
        pass

    def _gen_input():
        if input_fn is not None:
            x = input_fn()
        else:
            x = _make_input(input_shape, dev, dtype)
        return x

    sample = _gen_input()
    batch_size = (
        sample[0].shape[0] if isinstance(sample, tuple) else sample.shape[0]
    )

    with torch.inference_mode():
        # Warmup
        for _ in range(warmup):
            x = _gen_input()
            if isinstance(x, tuple):
                model(*x)
            else:
                model(x)
        _sync(dev)

        # Measure
        samples_ms: list[float] = []
        for _ in range(iters):
            x = _gen_input()
            _sync(dev)
            t0 = time.perf_counter()
            if isinstance(x, tuple):
                model(*x)
            else:
                model(x)
            _sync(dev)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.fmean(samples_ms)
    median_ms = statistics.median(samples_ms)
    std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
    throughput = (batch_size * 1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return BenchmarkResult(
        device=str(dev),
        dtype=str(dtype).replace("torch.", ""),
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
        mean_ms=mean_ms,
        median_ms=median_ms,
        std_ms=std_ms,
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        p50_ms=_percentile(samples_ms, 50),
        p90_ms=_percentile(samples_ms, 90),
        p99_ms=_percentile(samples_ms, 99),
        throughput_ips=throughput,
        samples_ms=samples_ms,
    )


def compare(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    input_shape: tuple[int, ...] | list[tuple[int, ...]],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    iters: int = 50,
) -> BenchmarkComparison:
    """Benchmark baseline and compressed models and return a comparison."""
    base = benchmark(
        baseline_model, input_shape, device=device, dtype=dtype,
        warmup=warmup, iters=iters,
    )
    comp = benchmark(
        compressed_model, input_shape, device=device, dtype=dtype,
        warmup=warmup, iters=iters,
    )
    return BenchmarkComparison(baseline=base, compressed=comp)
