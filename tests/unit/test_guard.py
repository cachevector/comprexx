"""Tests for AccuracyGuard."""

import warnings

import pytest

from comprexx.core.exceptions import AccuracyGuardTriggered
from comprexx.core.guard import AccuracyGuard


class TestAccuracyGuard:
    def test_within_threshold_no_raise(self):
        guard = AccuracyGuard(max_drop=0.02)
        # Drop of 1% is within 2% threshold
        guard.check(baseline=0.95, current=0.94, stage_name="test")

    def test_halt_on_exceed(self):
        guard = AccuracyGuard(max_drop=0.01, action="halt")
        with pytest.raises(AccuracyGuardTriggered) as exc_info:
            guard.check(baseline=0.95, current=0.90, stage_name="prune")
        assert exc_info.value.stage == "prune"
        assert exc_info.value.baseline == 0.95
        assert exc_info.value.current == 0.90

    def test_warn_on_exceed(self):
        guard = AccuracyGuard(max_drop=0.01, action="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.check(baseline=0.95, current=0.90, stage_name="quant")
            assert len(w) == 1
            assert "5.0%" in str(w[0].message)
            assert "quant" in str(w[0].message)

    def test_exact_threshold_no_raise(self):
        guard = AccuracyGuard(max_drop=0.05)
        # Exactly at threshold should not trigger (drop must exceed, not equal)
        guard.check(baseline=0.95, current=0.90, stage_name="test")

    def test_no_drop(self):
        guard = AccuracyGuard(max_drop=0.01)
        # Accuracy improved — no issue
        guard.check(baseline=0.90, current=0.92, stage_name="test")

    def test_custom_metric_name(self):
        guard = AccuracyGuard(metric="f1_score", max_drop=0.03)
        assert guard.metric == "f1_score"
