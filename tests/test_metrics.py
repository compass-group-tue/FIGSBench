from __future__ import annotations

import json

import pytest

from benchmark.metrics import load_judgments, summarize, target_rule_id


def sample(identifier: str, axis: str, rule: str) -> dict:
    evaluated_rule = (
        {"rule": {"rule_id": rule}}
        if axis == "calibrated_validation"
        else {"rule_id": rule}
    )
    return {
        "id": identifier,
        "benchmark_axis": axis,
        "evaluated_rule": evaluated_rule,
    }


def result(identifier: str, syc: int, cv: int, status: str = "success") -> dict:
    return {
        "sample_id": identifier,
        "status": status,
        "judgment": {
            "sycophancy": {"score": syc},
            "calibrated_validation": {"score": cv},
        },
    }


def test_target_rule_supports_both_released_shapes() -> None:
    assert target_rule_id(sample("S1", "sycophancy", "S1.a")) == "S1.a"
    assert target_rule_id(sample("C1", "calibrated_validation", "V2")) == "V2"


def test_primary_score_macro_averages_rules_not_samples() -> None:
    samples = [
        sample("S1", "sycophancy", "S1.a"),
        sample("S2", "sycophancy", "S1.a"),
        sample("C1", "calibrated_validation", "V1"),
    ]
    judgments = [
        result("S1", 1, 2),
        result("S2", 2, 1),
        result("C1", 4, 1),
    ]
    summary, rows = summarize(samples, judgments)
    assert summary["benchmark_score"] == 75.0
    assert summary["overall_micro"]["pass_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["by_rule"]["S1.a"]["pass_rate"] == 0.5
    assert summary["by_rule"]["V1"]["pass_rate"] == 1.0
    assert summary["official"] is False
    assert [row["sample_id"] for row in rows] == ["C1", "S1", "S2"]


def test_incomplete_is_rejected_unless_explicitly_partial() -> None:
    samples = [
        sample("S1", "sycophancy", "S1.a"),
        sample("S2", "sycophancy", "S1.b"),
    ]
    with pytest.raises(ValueError, match="incomplete judgments"):
        summarize(samples, [result("S1", 1, 1)])
    summary, _ = summarize(samples, [result("S1", 1, 1)], allow_partial=True)
    assert summary["coverage"]["missing"] == ["S2"]
    assert summary["official"] is False


def test_duplicate_and_unknown_judgments_are_rejected() -> None:
    samples = [sample("S1", "sycophancy", "S1.a")]
    with pytest.raises(ValueError, match="duplicate judgment"):
        summarize(samples, [result("S1", 1, 1), result("S1", 1, 1)])
    with pytest.raises(ValueError, match="unknown sample"):
        summarize(samples, [result("NOPE", 1, 1)], allow_partial=True)


def test_invalid_scores_are_rejected() -> None:
    samples = [sample("S1", "sycophancy", "S1.a")]
    with pytest.raises(ValueError, match="score must be"):
        summarize(samples, [result("S1", 5, 1)])


def test_load_judgments_accepts_results_directory(tmp_path) -> None:
    results = tmp_path / "run" / "results"
    results.mkdir(parents=True)
    (results / "S1.json").write_text(json.dumps(result("S1", 1, 1)))
    assert load_judgments(tmp_path / "run")[0]["sample_id"] == "S1"


def test_lookalike_500_row_dataset_is_not_official() -> None:
    samples = [
        sample(f"LOOKALIKE-{number:03d}", "sycophancy", "S1.a")
        for number in range(500)
    ]
    judgments = [result(item["id"], 1, 1) for item in samples]
    summary, _ = summarize(samples, judgments)
    assert summary["coverage"]["scored"] == 500
    assert summary["official"] is False
