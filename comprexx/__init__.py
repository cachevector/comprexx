"""Comprexx — ML Model Compression Toolkit."""

__version__ = "0.3.0"

from comprexx import stages
from comprexx.analysis.profiler import ModelProfile, analyze
from comprexx.analysis.sensitivity import (
    LayerSensitivity,
    SensitivityReport,
    analyze_sensitivity,
)
from comprexx.benchmark.runner import (
    BenchmarkComparison,
    BenchmarkResult,
    benchmark,
)
from comprexx.benchmark.runner import compare as compare_benchmarks
from comprexx.core.exceptions import (
    AccuracyGuardTriggered,
    CalibrationError,
    ComprexxError,
    ExportError,
    ModelLoadError,
    RecipeValidationError,
    UnsupportedLayerError,
)
from comprexx.core.guard import AccuracyGuard
from comprexx.core.pipeline import Pipeline, PipelineResult
from comprexx.core.report import CompressionReport, StageReport
from comprexx.export.onnx import ONNXExporter
from comprexx.recipe.loader import load_recipe

__all__ = [
    "AccuracyGuard",
    "AccuracyGuardTriggered",
    "BenchmarkComparison",
    "BenchmarkResult",
    "CalibrationError",
    "CompressionReport",
    "ComprexxError",
    "ExportError",
    "LayerSensitivity",
    "ModelLoadError",
    "ModelProfile",
    "SensitivityReport",
    "ONNXExporter",
    "Pipeline",
    "PipelineResult",
    "RecipeValidationError",
    "StageReport",
    "UnsupportedLayerError",
    "analyze",
    "analyze_sensitivity",
    "benchmark",
    "compare_benchmarks",
    "load_recipe",
    "stages",
]
