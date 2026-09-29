from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.release_builder import (
    ReleaseBuildError,
    assemble_release,
    build_release,
    canonicalize_ids,
    validate_expanded_records,
)


def sample(slot_id: str, axis: str, attempt: str) -> dict:
    return {
        "schema_version": 2,
        "id": slot_id,
        "benchmark_axis": axis,
        "scenario": {
            "id": slot_id + "-X4",
            "benchmark_axis": axis,
            "scenario_plan": f"unexpanded plan {attempt}",
            "nested": {"scenario_id": slot_id + "-X4"},
        },
        "original_transcript": {"scenario_id": slot_id + "-X4"},
        "attempt_marker": attempt,
    }


def result(slot_id: str, *, syc: int = 1, cv: int = 1) -> dict:
    return {
        "sample_id": slot_id,
        "status": "success",
        "judgment": {
            "axis_scored": "both",
            "sycophancy": {"score": syc},
            "calibrated_validation": {"score": cv},
        },
    }


def assemble(
    samples_a: list[dict],
    samples_b: list[dict],
    results_a: list[dict],
    results_b: list[dict],
    **kwargs,
):
    return assemble_release(
        attempt_a_samples=samples_a,
        attempt_b_samples=samples_b,
        attempt_a_results=results_a,
        attempt_b_results=results_b,
        gate_model="gate/model",
        **kwargs,
    )


def test_selects_larger_target_axis_score_and_orders_slots() -> None:
    syc_a = sample("FINAL-SYC-001", "sycophancy", "a")
    syc_b = sample("FINAL-SYC-001", "sycophancy", "b")
    cv_a = sample("FINAL-CV-001", "calibrated_validation", "a")
    cv_b = sample("FINAL-CV-001", "calibrated_validation", "b")
    release = assemble(
        [syc_a, cv_a],
        [cv_b, syc_b],
        [result("FINAL-SYC-001", syc=4), result("FINAL-CV-001", cv=1)],
        [result("FINAL-CV-001", cv=3), result("FINAL-SYC-001", syc=2)],
    )

    assert [record["id"] for record in release.records] == [
        "FINAL-CV-001",
        "FINAL-SYC-001",
    ]
    assert [record["attempt_marker"] for record in release.records] == ["b", "a"]
    assert release.gate_picks["FINAL-SYC-001"] == {
        "score_a": 4,
        "score_b": 2,
        "tied": False,
        "released_attempt": "a",
    }
    assert release.gate_manifest["axes"] == {
        "sycophancy": 1,
        "calibrated_validation": 1,
    }


def test_tie_winner_is_configurable_and_stored_picks_reproduce_history() -> None:
    a = sample("FINAL-CV-001", "calibrated_validation", "a")
    b = sample("FINAL-CV-001", "calibrated_validation", "b")
    tied = [result("FINAL-CV-001", cv=2)]
    default_b = assemble([a], [b], tied, tied, tie_winner="b")
    assert default_b.gate_picks["FINAL-CV-001"]["released_attempt"] == "b"

    saved = {
        "FINAL-CV-001": {
            "score_a": 2,
            "score_b": 2,
            "tied": True,
            "released_attempt": "b",
        }
    }
    historical = assemble(
        [a], [b], tied, tied, tie_winner="a", stored_picks=saved
    )
    assert historical.records[0]["attempt_marker"] == "b"
    assert historical.gate_manifest["tie_policy"]["stored_picks_applied"] is True


def test_stored_picks_cannot_override_a_strictly_harder_attempt() -> None:
    a = sample("FINAL-SYC-001", "sycophancy", "a")
    b = sample("FINAL-SYC-001", "sycophancy", "b")
    saved = {
        "FINAL-SYC-001": {
            "score_a": 4,
            "score_b": 1,
            "tied": False,
            "released_attempt": "b",
        }
    }
    with pytest.raises(ReleaseBuildError, match="cannot override strictly harder"):
        assemble(
            [a],
            [b],
            [result("FINAL-SYC-001", syc=4)],
            [result("FINAL-SYC-001", syc=1)],
            stored_picks=saved,
        )


@pytest.mark.parametrize(
    ("samples_b", "results_a", "message"),
    [
        (
            [sample("FINAL-SYC-002", "sycophancy", "b")],
            [result("FINAL-SYC-001")],
            "slot mismatch",
        ),
        (
            [sample("FINAL-CV-001", "calibrated_validation", "b")],
            [result("FINAL-SYC-001")],
            "slot mismatch",
        ),
    ],
)
def test_mismatched_attempt_or_judge_slots_are_rejected(
    samples_b: list[dict], results_a: list[dict], message: str
) -> None:
    a = sample("FINAL-SYC-001", "sycophancy", "a")
    with pytest.raises(ReleaseBuildError, match=message):
        assemble(
            [a],
            samples_b,
            results_a,
            [result(samples_b[0]["id"])],
        )


def test_axis_mismatch_and_duplicate_slots_are_rejected() -> None:
    wrong_axis = sample("FINAL-SYC-001", "calibrated_validation", "a")
    with pytest.raises(ReleaseBuildError, match="id implies sycophancy"):
        assemble(
            [wrong_axis],
            [wrong_axis],
            [result("FINAL-SYC-001")],
            [result("FINAL-SYC-001")],
        )

    duplicate = sample("FINAL-SYC-001", "sycophancy", "a")
    with pytest.raises(ReleaseBuildError, match="duplicate slot"):
        assemble(
            [duplicate, duplicate],
            [duplicate],
            [result("FINAL-SYC-001")],
            [result("FINAL-SYC-001")],
        )


def test_recursive_id_canonicalization_and_expanded_validation() -> None:
    raw = {
        "FINAL-CV-001-X4": [
            "CV-001",
            {"text": "references FINAL-CV-001-X4 here"},
        ]
    }
    assert canonicalize_ids(raw) == {
        "FINAL-CV-001": [
            "FINAL-CV-001",
            {"text": "references FINAL-CV-001 here"},
        ]
    }

    base = sample("FINAL-CV-001", "calibrated_validation", "a")
    expanded = json.loads(json.dumps(base))
    expanded["scenario"]["scenario_plan"] = "concrete expanded plan"
    expanded["scenario"]["expansion"] = "v4"
    expanded["scenario"]["id"] = "FINAL-CV-001-X4"
    validated = validate_expanded_records([base], [expanded])
    assert validated[0]["scenario"]["id"] == "FINAL-CV-001"
    assert validated[0]["scenario"]["scenario_plan"] == "concrete expanded plan"

    tampered = json.loads(json.dumps(expanded))
    tampered["attempt_marker"] = "tampered"
    with pytest.raises(ReleaseBuildError, match="fields other than"):
        validate_expanded_records([base], [tampered])


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def write_run(path: Path, records: list[dict]) -> Path:
    write_jsonl(path / "samples.jsonl", records)
    return path


def write_judge(path: Path, records: list[dict]) -> Path:
    results = path / "results"
    results.mkdir(parents=True)
    for record in records:
        (results / f"{record['sample_id']}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return path


def test_build_supports_split_runs_is_deterministic_and_requires_force(
    tmp_path: Path,
) -> None:
    cv_a = sample("FINAL-CV-001", "calibrated_validation", "a")
    syc_a = sample("FINAL-SYC-001", "sycophancy", "a")
    cv_b = sample("FINAL-CV-001", "calibrated_validation", "b")
    syc_b = sample("FINAL-SYC-001", "sycophancy", "b")
    a_runs = [
        write_run(tmp_path / "a-cv", [cv_a]),
        write_run(tmp_path / "a-syc", [syc_a]),
    ]
    b_runs = [
        write_run(tmp_path / "b-cv", [cv_b]),
        write_run(tmp_path / "b-syc", [syc_b]),
    ]
    judge_a = write_judge(
        tmp_path / "judge-a",
        [result("FINAL-CV-001", cv=1), result("FINAL-SYC-001", syc=4)],
    )
    judge_b = write_judge(
        tmp_path / "judge-b",
        [result("FINAL-CV-001", cv=3), result("FINAL-SYC-001", syc=2)],
    )

    kwargs = {
        "attempt_a_runs": a_runs,
        "attempt_b_runs": b_runs,
        "attempt_a_judge_runs": [judge_a],
        "attempt_b_judge_runs": [judge_b],
        "gate_model": "gate/model",
    }
    first = build_release(output_dir=tmp_path / "out-1", **kwargs)
    second = build_release(output_dir=tmp_path / "out-2", **kwargs)
    for name in ("gate_picks.json", "gate_manifest.json", "benchmark_unexpanded.jsonl"):
        assert (tmp_path / "out-1" / name).read_bytes() == (
            tmp_path / "out-2" / name
        ).read_bytes()

    manifest = json.loads(first.gate_manifest.read_text())
    assert manifest["outputs"]["gate_picks"]["sha256"]
    assert manifest["inputs"]["attempt_a_samples"][0]["sha256"]
    with pytest.raises(FileExistsError, match="without force"):
        build_release(output_dir=tmp_path / "out-1", **kwargs)
    build_release(output_dir=tmp_path / "out-1", force=True, **kwargs)


def test_build_validates_and_writes_an_existing_expansion(tmp_path: Path) -> None:
    a = sample("FINAL-CV-001", "calibrated_validation", "a")
    b = sample("FINAL-CV-001", "calibrated_validation", "b")
    run_a = write_run(tmp_path / "a", [a])
    run_b = write_run(tmp_path / "b", [b])
    judge_a = write_judge(tmp_path / "ja", [result("FINAL-CV-001", cv=3)])
    judge_b = write_judge(tmp_path / "jb", [result("FINAL-CV-001", cv=1)])

    expanded = json.loads(json.dumps(a))
    expanded["scenario"]["scenario_plan"] = "expanded, concrete plan"
    expanded["scenario"]["expansion"] = "offline-existing"
    # The selected release adds these gate fields before expansion.
    expanded["gate_winner_attempt"] = "a"
    expanded["gate_filter"] = "gate/model"
    expanded_path = tmp_path / "expanded.jsonl"
    write_jsonl(expanded_path, [expanded])

    artifacts = build_release(
        attempt_a_runs=[run_a],
        attempt_b_runs=[run_b],
        attempt_a_judge_runs=[judge_a],
        attempt_b_judge_runs=[judge_b],
        output_dir=tmp_path / "out",
        gate_model="gate/model",
        expanded_jsonl=expanded_path,
    )
    assert artifacts.benchmark_jsonl is not None
    published = json.loads(artifacts.benchmark_jsonl.read_text().strip())
    assert published["scenario"]["scenario_plan"] == "expanded, concrete plan"
    assert published["scenario"]["id"] == "FINAL-CV-001"

