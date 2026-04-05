"""Comprexx exception hierarchy."""

from __future__ import annotations


class ComprexxError(Exception):
    """Base exception for all Comprexx errors."""


class ModelLoadError(ComprexxError):
    """Failed to load or parse a model."""


class RecipeValidationError(ComprexxError):
    """Recipe YAML/config is invalid."""


class AccuracyGuardTriggered(ComprexxError):
    """Accuracy drop exceeded the configured threshold."""

    def __init__(self, stage: str, baseline: float, current: float, threshold: float) -> None:
        self.stage = stage
        self.baseline = baseline
        self.current = current
        self.threshold = threshold
        super().__init__(str(self))

    def __str__(self) -> str:
        drop = self.baseline - self.current
        drop_pct = drop * 100
        threshold_pct = self.threshold * 100
        return (
            f"Accuracy dropped {drop_pct:.1f}% after stage '{self.stage}', "
            f"exceeding the configured threshold of {threshold_pct:.1f}%.\n"
            f"\n"
            f"  Baseline:         {self.baseline * 100:.1f}%\n"
            f"  After stage:      {self.current * 100:.1f}%\n"
            f"  Configured limit: {threshold_pct:.1f}% drop max\n"
            f"\n"
            f"  Suggestions:\n"
            f"  -> Exclude sensitive layers via exclude_layers\n"
            f"  -> Use a less aggressive compression setting\n"
            f"  -> Add a QAT or distillation recovery stage\n"
            f"  -> Run sensitivity analysis to identify problematic layers"
        )


class ExportError(ComprexxError):
    """Failed to export model to target format."""


class CalibrationError(ComprexxError):
    """Calibration data missing or insufficient."""


class UnsupportedLayerError(ComprexxError):
    """Model contains layers not supported by the requested operation."""
