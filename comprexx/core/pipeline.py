"""Pipeline engine — orchestrates compression stages."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from comprexx.analysis.profiler import ModelProfile, analyze
from comprexx.core.guard import AccuracyGuard
from comprexx.core.report import CompressionReport, StageReport
from comprexx.stages.base import CompressionStage, StageContext


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    model: nn.Module
    report: CompressionReport
    run_dir: str
    profile_before: ModelProfile
    profile_after: ModelProfile

    def summary(self) -> str:
        return self.report.summary()


class Pipeline:
    """Ordered sequence of compression stages applied to a model.

    Supports dry-run mode, accuracy guards, and automatic run directory creation.
    """

    def __init__(self, stages: list[CompressionStage]):
        self.stages = stages

    def run(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        calibration_data: Optional[DataLoader] = None,
        eval_fn: Optional[Callable[[nn.Module], dict[str, float]]] = None,
        accuracy_guard: Optional[AccuracyGuard] = None,
        dry_run: bool = False,
        device: str = "cpu",
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """Run the compression pipeline.

        Args:
            model: PyTorch model to compress.
            input_shape: Model input shape (including batch dim).
            calibration_data: DataLoader for calibration (needed by PTQ static).
            eval_fn: Callable that takes model and returns {metric: value} dict.
            accuracy_guard: Accuracy threshold configuration.
            dry_run: If True, estimate only without applying compression.
            device: Device to run on.
            output_dir: Base directory for run artifacts.

        Returns:
            PipelineResult with compressed model, report, and run directory.
        """
        pipeline_start = time.time()

        # Create run directory
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model.__class__.__name__}"
        base = Path(output_dir) if output_dir else Path("./comprexx_runs")
        run_dir = base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stage_reports").mkdir(exist_ok=True)

        # Profile original model
        model_name = model.__class__.__name__
        profile_before = analyze(model, input_shape, device, model_name=model_name)
        profile_before.save(run_dir / "model_profile.json")

        # Get baseline accuracy
        baseline_accuracy: Optional[float] = None
        if eval_fn is not None:
            metrics = eval_fn(model)
            if accuracy_guard is not None:
                baseline_accuracy = metrics.get(accuracy_guard.metric)

        # Build stage context
        context = StageContext(
            input_shape=input_shape,
            device=device,
            calibration_data=calibration_data,
            eval_fn=eval_fn,
        )

        # Execute stages
        stage_reports: list[StageReport] = []
        current_model = model

        for i, stage in enumerate(self.stages):
            if dry_run:
                estimate = stage.estimate(profile_before)
                report = StageReport(
                    stage_name=stage.name,
                    technique=f"{stage.name}_estimate",
                    duration_seconds=0.0,
                    param_count_before=profile_before.total_params,
                    param_count_after=int(
                        profile_before.total_params
                        * (1 - estimate.estimated_size_reduction_pct / 100)
                    ),
                    flops_before=profile_before.total_flops,
                    flops_after=int(
                        profile_before.total_flops
                        * (1 - estimate.estimated_flops_reduction_pct / 100)
                    ),
                    size_bytes_before=profile_before.size_bytes,
                    size_bytes_after=int(
                        profile_before.size_bytes
                        * (1 - estimate.estimated_size_reduction_pct / 100)
                    ),
                    notes=estimate.notes,
                )
            else:
                current_model, report = stage.apply(current_model, context)

                # Evaluate accuracy after stage
                if eval_fn is not None:
                    metrics = eval_fn(current_model)
                    if accuracy_guard is not None and baseline_accuracy is not None:
                        current_accuracy = metrics.get(accuracy_guard.metric)
                        if current_accuracy is not None:
                            report.accuracy_before = baseline_accuracy
                            report.accuracy_after = current_accuracy
                            report.accuracy_delta = current_accuracy - baseline_accuracy

                            # Check guard
                            accuracy_guard.check(
                                baseline_accuracy, current_accuracy, stage.name
                            )

            # Save stage report
            stage_idx = f"{i + 1:02d}"
            report_path = run_dir / "stage_reports" / f"{stage_idx}_{stage.name}.json"
            report_path.write_text(report.to_json())
            stage_reports.append(report)

        # Profile final model
        if not dry_run and stage_reports:
            profile_after = analyze(current_model, input_shape, device, model_name=model_name)
        else:
            profile_after = profile_before

        # Build aggregate report
        total_duration = time.time() - pipeline_start
        compression_report = CompressionReport(
            model_name=model_name,
            stages=stage_reports,
            total_duration_seconds=total_duration,
        )
        compression_report.save(run_dir / "compression_report.json")

        return PipelineResult(
            model=current_model,
            report=compression_report,
            run_dir=str(run_dir),
            profile_before=profile_before,
            profile_after=profile_after,
        )
