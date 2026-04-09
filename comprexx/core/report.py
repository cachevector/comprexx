"""Compression reports: per-stage and aggregate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StageReport:
    """Report produced by a single compression stage."""

    stage_name: str
    technique: str
    duration_seconds: float
    param_count_before: int
    param_count_after: int
    flops_before: int
    flops_after: int
    size_bytes_before: int
    size_bytes_after: int
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    accuracy_delta: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.size_bytes_after == 0:
            return float("inf")
        return self.size_bytes_before / self.size_bytes_after

    @property
    def flops_reduction_pct(self) -> float:
        if self.flops_before == 0:
            return 0.0
        return (1 - self.flops_after / self.flops_before) * 100

    @property
    def size_reduction_pct(self) -> float:
        if self.size_bytes_before == 0:
            return 0.0
        return (1 - self.size_bytes_after / self.size_bytes_before) * 100

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compression_ratio"] = self.compression_ratio
        d["flops_reduction_pct"] = self.flops_reduction_pct
        d["size_reduction_pct"] = self.size_reduction_pct
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        lines = [
            f"Stage: {self.stage_name} ({self.technique})",
            f"  Duration:    {self.duration_seconds:.2f}s",
            f"  Params:      {self.param_count_before:,} -> {self.param_count_after:,}",
            f"  Size:        {self.size_bytes_before / 1e6:.2f} MB -> "
            f"{self.size_bytes_after / 1e6:.2f} MB ({self.size_reduction_pct:.1f}% reduction)",
            f"  FLOPs:       {self.flops_before:,} -> {self.flops_after:,} "
            f"({self.flops_reduction_pct:.1f}% reduction)",
        ]
        if self.accuracy_before is not None and self.accuracy_after is not None:
            lines.append(
                f"  Accuracy:    {self.accuracy_before * 100:.2f}% -> "
                f"{self.accuracy_after * 100:.2f}% "
                f"(delta: {self.accuracy_delta * 100:+.2f}%)"
            )
        if self.notes:
            lines.append(f"  Notes:       {'; '.join(self.notes)}")
        return "\n".join(lines)


@dataclass
class CompressionReport:
    """Aggregate report for a full compression pipeline run."""

    model_name: str
    stages: list[StageReport] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def total_compression_ratio(self) -> float:
        if not self.stages:
            return 1.0
        first = self.stages[0]
        last = self.stages[-1]
        if last.size_bytes_after == 0:
            return float("inf")
        return first.size_bytes_before / last.size_bytes_after

    @property
    def total_flops_reduction_pct(self) -> float:
        if not self.stages:
            return 0.0
        first = self.stages[0]
        last = self.stages[-1]
        if first.flops_before == 0:
            return 0.0
        return (1 - last.flops_after / first.flops_before) * 100

    @property
    def total_size_reduction_pct(self) -> float:
        if not self.stages:
            return 0.0
        first = self.stages[0]
        last = self.stages[-1]
        if first.size_bytes_before == 0:
            return 0.0
        return (1 - last.size_bytes_after / first.size_bytes_before) * 100

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_duration_seconds": self.total_duration_seconds,
            "total_compression_ratio": self.total_compression_ratio,
            "total_flops_reduction_pct": self.total_flops_reduction_pct,
            "total_size_reduction_pct": self.total_size_reduction_pct,
            "stages": [s.to_dict() for s in self.stages],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    def summary(self) -> str:
        lines = [
            f"Compression Report: {self.model_name}",
            f"{'=' * 50}",
            f"  Total duration: {self.total_duration_seconds:.2f}s",
            f"  Stages:         {len(self.stages)}",
            f"  Compression:    {self.total_compression_ratio:.2f}x",
            f"  Size reduction: {self.total_size_reduction_pct:.1f}%",
            f"  FLOPs reduction:{self.total_flops_reduction_pct:.1f}%",
            "",
        ]
        for stage in self.stages:
            lines.append(stage.summary())
            lines.append("")
        return "\n".join(lines)
