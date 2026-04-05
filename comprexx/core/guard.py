"""Accuracy guard — halts or warns when accuracy drops past threshold."""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel

from comprexx.core.exceptions import AccuracyGuardTriggered


class AccuracyGuard(BaseModel):
    """Configurable accuracy threshold for compression pipelines."""

    metric: str = "top1_accuracy"
    max_drop: float = 0.02
    action: Literal["halt", "warn"] = "halt"

    def check(self, baseline: float, current: float, stage_name: str) -> None:
        """Check if accuracy drop exceeds threshold.

        Raises AccuracyGuardTriggered if action is 'halt'.
        Emits a warning if action is 'warn'.
        """
        drop = baseline - current
        if drop > self.max_drop:
            if self.action == "halt":
                raise AccuracyGuardTriggered(
                    stage=stage_name,
                    baseline=baseline,
                    current=current,
                    threshold=self.max_drop,
                )
            else:
                warnings.warn(
                    f"Accuracy dropped {drop * 100:.1f}% after stage '{stage_name}' "
                    f"(threshold: {self.max_drop * 100:.1f}%)",
                    stacklevel=2,
                )
