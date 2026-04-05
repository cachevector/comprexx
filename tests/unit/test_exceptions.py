"""Tests for the exception hierarchy."""

import pytest

from comprexx.core.exceptions import (
    AccuracyGuardTriggered,
    CalibrationError,
    ComprexxError,
    ExportError,
    ModelLoadError,
    RecipeValidationError,
    UnsupportedLayerError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from ComprexxError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ModelLoadError,
            RecipeValidationError,
            ExportError,
            CalibrationError,
            UnsupportedLayerError,
        ],
    )
    def test_subclass_of_comprexx_error(self, exc_cls):
        exc = exc_cls("test message")
        assert isinstance(exc, ComprexxError)
        assert str(exc) == "test message"

    def test_accuracy_guard_triggered(self):
        exc = AccuracyGuardTriggered(
            stage="ptq_static", baseline=0.761, current=0.729, threshold=0.01
        )
        assert isinstance(exc, ComprexxError)
        assert exc.stage == "ptq_static"
        assert exc.baseline == 0.761
        assert exc.current == 0.729
        assert exc.threshold == 0.01

    def test_accuracy_guard_triggered_message(self):
        exc = AccuracyGuardTriggered(
            stage="ptq_static", baseline=0.761, current=0.729, threshold=0.01
        )
        msg = str(exc)
        assert "3.2%" in msg
        assert "ptq_static" in msg
        assert "1.0%" in msg
        assert "76.1%" in msg
        assert "72.9%" in msg

    def test_accuracy_guard_triggered_catchable_as_base(self):
        with pytest.raises(ComprexxError):
            raise AccuracyGuardTriggered(
                stage="prune", baseline=0.90, current=0.80, threshold=0.05
            )
