"""Tests for StageReport and CompressionReport."""

import json

from comprexx.core.report import CompressionReport, StageReport


def _make_stage_report(**overrides) -> StageReport:
    defaults = dict(
        stage_name="test_stage",
        technique="test_technique",
        duration_seconds=1.5,
        param_count_before=1000,
        param_count_after=700,
        flops_before=2_000_000,
        flops_after=1_400_000,
        size_bytes_before=4000,
        size_bytes_after=2800,
    )
    defaults.update(overrides)
    return StageReport(**defaults)


class TestStageReport:
    def test_compression_ratio(self):
        r = _make_stage_report(size_bytes_before=4000, size_bytes_after=2000)
        assert r.compression_ratio == 2.0

    def test_compression_ratio_zero_after(self):
        r = _make_stage_report(size_bytes_after=0)
        assert r.compression_ratio == float("inf")

    def test_flops_reduction_pct(self):
        r = _make_stage_report(flops_before=1000, flops_after=700)
        assert abs(r.flops_reduction_pct - 30.0) < 0.01

    def test_size_reduction_pct(self):
        r = _make_stage_report(size_bytes_before=1000, size_bytes_after=600)
        assert abs(r.size_reduction_pct - 40.0) < 0.01

    def test_to_dict_contains_computed(self):
        r = _make_stage_report()
        d = r.to_dict()
        assert "compression_ratio" in d
        assert "flops_reduction_pct" in d
        assert "size_reduction_pct" in d

    def test_to_json_roundtrip(self):
        r = _make_stage_report()
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["stage_name"] == "test_stage"
        assert parsed["technique"] == "test_technique"

    def test_summary_contains_key_info(self):
        r = _make_stage_report()
        s = r.summary()
        assert "test_stage" in s
        assert "test_technique" in s

    def test_summary_with_accuracy(self):
        r = _make_stage_report(
            accuracy_before=0.95, accuracy_after=0.93, accuracy_delta=-0.02
        )
        s = r.summary()
        assert "95.00%" in s
        assert "93.00%" in s


class TestCompressionReport:
    def test_empty_report(self):
        r = CompressionReport(model_name="test_model")
        assert r.total_compression_ratio == 1.0
        assert r.total_flops_reduction_pct == 0.0
        assert r.total_size_reduction_pct == 0.0

    def test_aggregate_metrics(self):
        s1 = _make_stage_report(
            stage_name="s1",
            size_bytes_before=4000,
            size_bytes_after=3000,
            flops_before=2000,
            flops_after=1500,
        )
        s2 = _make_stage_report(
            stage_name="s2",
            size_bytes_before=3000,
            size_bytes_after=1500,
            flops_before=1500,
            flops_after=1000,
        )
        r = CompressionReport(model_name="test", stages=[s1, s2])
        # Total: 4000 -> 1500
        assert abs(r.total_compression_ratio - 4000 / 1500) < 0.01
        assert abs(r.total_size_reduction_pct - 62.5) < 0.1
        # FLOPs: 2000 -> 1000
        assert abs(r.total_flops_reduction_pct - 50.0) < 0.01

    def test_to_json(self):
        s1 = _make_stage_report(stage_name="s1")
        r = CompressionReport(model_name="test", stages=[s1], total_duration_seconds=2.0)
        parsed = json.loads(r.to_json())
        assert parsed["model_name"] == "test"
        assert len(parsed["stages"]) == 1

    def test_save(self, tmp_path):
        r = CompressionReport(model_name="test", stages=[_make_stage_report()])
        out = tmp_path / "report.json"
        r.save(out)
        parsed = json.loads(out.read_text())
        assert parsed["model_name"] == "test"

    def test_summary(self):
        r = CompressionReport(
            model_name="resnet50",
            stages=[_make_stage_report()],
            total_duration_seconds=5.0,
        )
        s = r.summary()
        assert "resnet50" in s
        assert "5.00s" in s
