"""Base classes for compression stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from comprexx.core.report import StageReport

if TYPE_CHECKING:
    from comprexx.analysis.profiler import ModelProfile


@dataclass
class StageContext:
    """Context passed to each compression stage during pipeline execution."""

    input_shape: tuple[int, ...]
    device: str = "cpu"
    calibration_data: Optional[DataLoader] = None
    eval_fn: Optional[Callable[[nn.Module], dict[str, float]]] = None


@dataclass
class StageEstimate:
    """Estimated impact of a compression stage (for dry-run mode)."""

    estimated_compression_ratio: float = 1.0
    estimated_flops_reduction_pct: float = 0.0
    estimated_size_reduction_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


class CompressionStage(ABC):
    """Abstract base class for all compression stages.

    Every compression technique (quantization, pruning, etc.) implements this
    interface so it can be composed into a Pipeline.
    """

    name: str = "base_stage"

    @abstractmethod
    def apply(
        self, model: nn.Module, context: StageContext
    ) -> tuple[nn.Module, StageReport]:
        """Apply compression to the model.

        Returns:
            A tuple of (compressed_model, stage_report).
        """

    def estimate(self, profile: ModelProfile) -> StageEstimate:
        """Estimate compression impact without applying. Override in subclasses."""
        return StageEstimate(notes=["No estimate available for this stage."])
