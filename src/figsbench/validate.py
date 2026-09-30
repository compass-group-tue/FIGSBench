"""Check the FIGS data files, the selection record and the frozen judge prompts.

It uses only the Python standard library, so it runs before any model-serving
or API dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AXES = ("sycophancy", "calibrated_validation")
EXPECTED_AXIS_COUNTS = Counter({"sycophancy": 390, "calibrated_validation": 110})
EXPECTED_RULE_COUNTS = Counter(
    {
        "S1.a": 54,
        "S1.b": 55,
        "S1.c": 58,
        "S2.a": 56,
        "S2.b": 57,
        "S2.c": 54,
        "S2.d": 56,
        "V1": 37,
        "V2": 36,
        "V3": 37,
    }
)
EXPECTED_IDS = frozenset(
    [f"FINAL-SYC-{number:03d}" for number in range(1, 391)]
    + [f"FINAL-CV-{number:03d}" for number in range(1, 111)]
)
ID_PATTERN = re.compile(r"^FINAL-(SYC|CV)-\d{3}$")
HEX_256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationIssue:
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


class SchemaValidationError(ValueError):
    """Raised with every detected validation issue, not merely the first."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        preview = "\n".join(f"- {issue}" for issue in self.issues[:25])
        remainder = len(self.issues) - 25
        if remainder > 0:
            preview += f"\n- ... and {remainder} more issue(s)"
        super().__init__(f"release validation failed ({len(self.issues)} issue(s))\n{preview}")


def _issue(issues: list[ValidationIssue], location: str, message: str) -> None:
    issues.append(ValidationIssue(location, message))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
        value, float
    )


def _nonempty_string(
    value: Any, issues: list[ValidationIssue], location: str
) -> bool:
    if not isinstance(value, str) or not value.strip():
        _issue(issues, location, "must be a non-empty string")
        return False
    return True


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(
            [ValidationIssue(str(source), f"cannot read valid JSON: {exc}")]
        ) from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SchemaValidationError(
            [ValidationIssue(str(source), f"cannot read JSONL: {exc}")]
        ) from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _issue(issues, f"{source}:{line_number}", f"invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            _issue(issues, f"{source}:{line_number}", "row must be a JSON object")
            continue
        records.append(value)
    if issues:
        raise SchemaValidationError(issues)
    return records


def target_rule_id(sample: Mapping[str, Any]) -> str | None:
    evaluated = sample.get("evaluated_rule")
    if not isinstance(evaluated, Mapping):
        return None
    rule = evaluated.get("rule")
    value = rule if isinstance(rule, Mapping) else evaluated
    rule_id = value.get("rule_id")
    if isinstance(rule_id, str) and rule_id:
        return rule_id
    primary = value.get("primary_rule_ids")
    if isinstance(primary, list) and len(primary) == 1 and isinstance(primary[0], str):
        return primary[0]
    return None


def _validate_named_entity(
    value: Any,
    issues: list[ValidationIssue],
    location: str,
    *,
    require_confidence: bool = False,
) -> None:
    if not isinstance(value, Mapping):
        _issue(issues, location, "must be an object")
        return
    _nonempty_string(value.get("id"), issues, f"{location}.id")
    _nonempty_string(value.get("name"), issues, f"{location}.name")
    if require_confidence:
        confidence = value.get("confidence")
        # Historical authoring records use null when a deterministic domain
        # assignment did not produce a calibrated confidence value.
        if confidence is not None and (
            not _is_number(confidence) or not 0 <= confidence <= 1
        ):
            _issue(issues, f"{location}.confidence", "must be a number from 0 to 1")


def _validate_rule(
    evaluated: Any,
    axis: Any,
    issues: list[ValidationIssue],
    location: str,
) -> str | None:
    if not isinstance(evaluated, Mapping):
        _issue(issues, location, "must be an object")
        return None
    nested = evaluated.get("rule")
    rule = nested if isinstance(nested, Mapping) else evaluated
    rule_id = rule.get("rule_id")
    if not _nonempty_string(rule_id, issues, f"{location}.rule_id"):
        return None
    allowed = {"S1.a", "S1.b", "S1.c", "S2.a", "S2.b", "S2.c", "S2.d"}
    if axis == "calibrated_validation":
        allowed = {"V1", "V2", "V3"}
    if rule_id not in allowed:
        _issue(issues, f"{location}.rule_id", f"{rule_id!r} is not valid for axis {axis!r}")
    for field in ("chunk_id", "title", "description", "example", "neighboring_boundary"):
        _nonempty_string(rule.get(field), issues, f"{location}.{field}")
    if rule.get("axis") != axis:
        _issue(issues, f"{location}.axis", "must match the sample benchmark_axis")
    primary = rule.get("primary_rule_ids")
    if primary != [rule_id]:
        _issue(issues, f"{location}.primary_rule_ids", "must contain exactly the target rule")
    if isinstance(nested, Mapping):
        if evaluated.get("axis") != axis:
            _issue(issues, f"{location}.axis", "outer rule axis must match the sample")
        if evaluated.get("primary_rule_ids") != [rule_id]:
            _issue(
                issues,
                f"{location}.primary_rule_ids",
                "outer rule must contain exactly the target rule",
            )
    return str(rule_id)


def _validate_transcript(
    transcript: Any,
    sample: Mapping[str, Any],
    evaluated_rule_id: str | None,
    issues: list[ValidationIssue],
    location: str,
) -> None:
    if not isinstance(transcript, Mapping):
        _issue(issues, location, "must be an object")
        return
    if transcript.get("schema_version") != 2:
        _issue(issues, f"{location}.schema_version", "must equal 2")
    comparisons = {
        "scenario_id": sample.get("id"),
        "benchmark_axis": sample.get("benchmark_axis"),
        "source_seed_id": sample.get("source_seed_id"),
        "texting_style_id": (
            sample.get("texting_style", {}).get("id")
            if isinstance(sample.get("texting_style"), Mapping)
            else None
        ),
        # transcript.evaluated_rule_id holds the rule's chunk_id (for example ``S1.a``).
        "evaluated_rule_id": evaluated_rule_id,
    }
    for field, expected in comparisons.items():
        if transcript.get(field) != expected:
            _issue(issues, f"{location}.{field}", f"must equal {expected!r}")
    fingerprint = transcript.get("scenario_fingerprint")
    if not isinstance(fingerprint, str) or not HEX_256_PATTERN.fullmatch(fingerprint):
        _issue(issues, f"{location}.scenario_fingerprint", "must be a lowercase SHA-256")
    for field in ("purpose", "assistant_model", "user_model"):
        _nonempty_string(transcript.get(field), issues, f"{location}.{field}")
    if not isinstance(transcript.get("first_user_turn_generated"), bool):
        _issue(issues, f"{location}.first_user_turn_generated", "must be boolean")

    turns = transcript.get("turns")
    if not isinstance(turns, list):
        _issue(issues, f"{location}.turns", "must be an array")
        return
    if len(turns) != 10:
        _issue(issues, f"{location}.turns", f"must contain exactly 10 turns, found {len(turns)}")
    for index, turn in enumerate(turns):
        turn_location = f"{location}.turns[{index}]"
        if not isinstance(turn, Mapping):
            _issue(issues, turn_location, "must be an object")
            continue
        expected_number = index + 1
        expected_role = "user" if index % 2 == 0 else "assistant"
        if turn.get("turn") != expected_number:
            _issue(issues, f"{turn_location}.turn", f"must equal {expected_number}")
        if turn.get("role") != expected_role:
            _issue(issues, f"{turn_location}.role", f"must equal {expected_role!r}")
        _nonempty_string(turn.get("content"), issues, f"{turn_location}.content")


def _validate_sample(
    sample: Any,
    index: int,
    issues: list[ValidationIssue],
    *,
    canonical: bool,
) -> None:
    location = f"samples[{index}]"
    if not isinstance(sample, Mapping):
        _issue(issues, location, "must be an object")
        return
    required = {
        "schema_version",
        "id",
        "benchmark_axis",
        "domain",
        "evaluated_rule",
        "source_seed_id",
        "texting_style",
        "scenario",
        "original_transcript",
        "transcript",
        "scenario_refinement_iterations",
        "scenario_refinement_history",
        "audit_iterations",
        "audit_history",
        "terminal_audit",
        "terminal_audit_transcript_fingerprint",
        "models",
        "gate_winner_attempt",
        "gate_filter",
    }
    for field in sorted(required - set(sample)):
        _issue(issues, f"{location}.{field}", "required field is missing")
    if sample.get("schema_version") != 2:
        _issue(issues, f"{location}.schema_version", "must equal 2")

    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not ID_PATTERN.fullmatch(sample_id):
        _issue(issues, f"{location}.id", "must match FINAL-SYC-NNN or FINAL-CV-NNN")
    axis = sample.get("benchmark_axis")
    if axis not in AXES:
        _issue(issues, f"{location}.benchmark_axis", f"must be one of {AXES}")
    elif isinstance(sample_id, str):
        expected_prefix = "FINAL-SYC-" if axis == "sycophancy" else "FINAL-CV-"
        if not sample_id.startswith(expected_prefix):
            _issue(issues, f"{location}.id", f"must start with {expected_prefix} for this axis")

    _validate_named_entity(sample.get("domain"), issues, f"{location}.domain", require_confidence=True)
    _validate_named_entity(sample.get("texting_style"), issues, f"{location}.texting_style")
    style = sample.get("texting_style")
    if isinstance(style, Mapping):
        _nonempty_string(style.get("description"), issues, f"{location}.texting_style.description")

    source_seed_id = sample.get("source_seed_id")
    if not isinstance(source_seed_id, str) or not HEX_256_PATTERN.fullmatch(source_seed_id):
        _issue(issues, f"{location}.source_seed_id", "must be a lowercase SHA-256 identifier")
    rule_id = _validate_rule(sample.get("evaluated_rule"), axis, issues, f"{location}.evaluated_rule")
    evaluated = sample.get("evaluated_rule")
    evaluated_rule_id = (
        evaluated.get("chunk_id") if isinstance(evaluated, Mapping) else None
    )
    if not isinstance(evaluated_rule_id, str) or not evaluated_rule_id:
        evaluated_rule_id = rule_id

    scenario = sample.get("scenario")
    if not isinstance(scenario, Mapping):
        _issue(issues, f"{location}.scenario", "must be an object")
    else:
        for field in (
            "user_role",
            "scenario_plan",
            "archetype_fit_rationale",
            "rule_elicitation_rationale",
        ):
            _nonempty_string(scenario.get(field), issues, f"{location}.scenario.{field}")
        if canonical and scenario.get("expansion") != "expansion-v1":
            _issue(
                issues,
                f"{location}.scenario.expansion",
                "canonical release must use 'expansion-v1'",
            )
        for field in ("benchmark_axis", "domain", "evaluated_rule", "source_seed_id"):
            if scenario.get(field) != sample.get(field):
                _issue(issues, f"{location}.scenario.{field}", "must match the top-level value")

    _validate_transcript(
        sample.get("original_transcript"),
        sample,
        evaluated_rule_id,
        issues,
        f"{location}.original_transcript",
    )
    _validate_transcript(
        sample.get("transcript"), sample, evaluated_rule_id, issues, f"{location}.transcript"
    )

    for count_field, history_field in (
        ("scenario_refinement_iterations", "scenario_refinement_history"),
        ("audit_iterations", "audit_history"),
    ):
        count = sample.get(count_field)
        history = sample.get(history_field)
        if not _is_int(count) or count < 0:
            _issue(issues, f"{location}.{count_field}", "must be a non-negative integer")
        if not isinstance(history, list):
            _issue(issues, f"{location}.{history_field}", "must be an array")
        elif _is_int(count) and len(history) != count:
            _issue(
                issues,
                f"{location}.{history_field}",
                f"length must equal {count_field} ({count})",
            )
        if canonical and count != 5:
            _issue(issues, f"{location}.{count_field}", "canonical release must record 5 iterations")

    if not isinstance(sample.get("terminal_audit"), Mapping):
        _issue(issues, f"{location}.terminal_audit", "must be an object")
    terminal_fingerprint = sample.get("terminal_audit_transcript_fingerprint")
    if not isinstance(terminal_fingerprint, str) or not HEX_256_PATTERN.fullmatch(
        terminal_fingerprint
    ):
        _issue(
            issues,
            f"{location}.terminal_audit_transcript_fingerprint",
            "must be a lowercase SHA-256",
        )
    models = sample.get("models")
    if not isinstance(models, Mapping):
        _issue(issues, f"{location}.models", "must be an object")
    else:
        for role in ("generator", "refiner", "auditor", "assistant", "user"):
            _nonempty_string(models.get(role), issues, f"{location}.models.{role}")
    if sample.get("gate_winner_attempt") not in {"a", "b"}:
        _issue(issues, f"{location}.gate_winner_attempt", "must be 'a' or 'b'")
    _nonempty_string(sample.get("gate_filter"), issues, f"{location}.gate_filter")


def validate_dataset(
    samples: Sequence[Mapping[str, Any]], *, canonical: bool = True
) -> dict[str, Any]:
    """Validate records and return a deterministic composition summary."""

    issues: list[ValidationIssue] = []
    for index, sample in enumerate(samples):
        _validate_sample(sample, index, issues, canonical=canonical)

    identifiers = [sample.get("id") for sample in samples if isinstance(sample, Mapping)]
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        _issue(issues, "samples", f"duplicate IDs: {duplicates[:20]}")

    axis_counts = Counter(
        sample.get("benchmark_axis") for sample in samples if isinstance(sample, Mapping)
    )
    rule_counts = Counter(
        target_rule_id(sample) for sample in samples if isinstance(sample, Mapping)
    )
    if canonical:
        if len(samples) != 500:
            _issue(issues, "samples", f"canonical release must contain 500 rows, found {len(samples)}")
        actual_ids = set(identifier for identifier in identifiers if isinstance(identifier, str))
        missing = sorted(EXPECTED_IDS - actual_ids)
        unexpected = sorted(actual_ids - EXPECTED_IDS)
        if missing:
            _issue(issues, "samples", f"missing canonical IDs: {missing[:20]}")
        if unexpected:
            _issue(issues, "samples", f"unexpected IDs: {unexpected[:20]}")
        if axis_counts != EXPECTED_AXIS_COUNTS:
            _issue(
                issues,
                "samples",
                f"axis counts must be {dict(EXPECTED_AXIS_COUNTS)}, found {dict(axis_counts)}",
            )
        if rule_counts != EXPECTED_RULE_COUNTS:
            _issue(
                issues,
                "samples",
                f"rule counts must be {dict(EXPECTED_RULE_COUNTS)}, found {dict(rule_counts)}",
            )

    if issues:
        raise SchemaValidationError(issues)
    return {
        "samples": len(samples),
        "axes": dict(sorted(axis_counts.items(), key=lambda item: str(item[0]))),
        "rules": dict(sorted(rule_counts.items(), key=lambda item: str(item[0]))),
    }


def validate_gate(
    samples: Sequence[Mapping[str, Any]],
    picks: Any,
    manifest: Any,
) -> dict[str, Any]:
    issues: list[ValidationIssue] = []
    sample_by_id = {
        sample.get("id"): sample
        for sample in samples
        if isinstance(sample, Mapping) and isinstance(sample.get("id"), str)
    }
    if not isinstance(picks, Mapping):
        raise SchemaValidationError([ValidationIssue("gate_picks", "must be an object")])
    if set(picks) != set(sample_by_id):
        missing = sorted(set(sample_by_id) - set(picks))
        unexpected = sorted(set(picks) - set(sample_by_id))
        if missing:
            _issue(issues, "gate_picks", f"missing sample IDs: {missing[:20]}")
        if unexpected:
            _issue(issues, "gate_picks", f"unknown sample IDs: {unexpected[:20]}")

    tied = 0
    released_counts: Counter[str] = Counter()
    for sample_id, value in picks.items():
        location = f"gate_picks.{sample_id}"
        if not isinstance(value, Mapping):
            _issue(issues, location, "must be an object")
            continue
        score_a, score_b = value.get("score_a"), value.get("score_b")
        for field, score in (("score_a", score_a), ("score_b", score_b)):
            if not _is_int(score) or not 1 <= score <= 4:
                _issue(issues, f"{location}.{field}", "must be an integer from 1 to 4")
        is_tied = _is_int(score_a) and _is_int(score_b) and score_a == score_b
        if value.get("tied") is not is_tied:
            _issue(issues, f"{location}.tied", "must exactly reflect score equality")
        tied += int(is_tied)
        released = value.get("released_attempt")
        if released not in {"a", "b"}:
            _issue(issues, f"{location}.released_attempt", "must be 'a' or 'b'")
            continue
        released_counts[released] += 1
        if not is_tied and _is_int(score_a) and _is_int(score_b):
            expected = "a" if score_a > score_b else "b"
            if released != expected:
                _issue(
                    issues,
                    f"{location}.released_attempt",
                    "must select the higher (harder/worse) target-axis score",
                )
        sample = sample_by_id.get(sample_id)
        if sample is not None and sample.get("gate_winner_attempt") != released:
            _issue(issues, f"{location}.released_attempt", "does not match canonical sample")

    if not isinstance(manifest, Mapping):
        _issue(issues, "gate_manifest", "must be an object")
    else:
        expected_values = {
            "benchmark": "FIGS",
            "n": len(samples),
            "axes": dict(EXPECTED_AXIS_COUNTS),
            "tied": tied,
            "released_attempt_counts": dict(released_counts),
        }
        for field, expected in expected_values.items():
            if manifest.get(field) != expected:
                _issue(
                    issues,
                    f"gate_manifest.{field}",
                    f"must equal {expected!r}, found {manifest.get(field)!r}",
                )
        _nonempty_string(manifest.get("gate_model"), issues, "gate_manifest.gate_model")
        _nonempty_string(manifest.get("gate"), issues, "gate_manifest.gate")

    if issues:
        raise SchemaValidationError(issues)
    return {"records": len(picks), "tied": tied, "released_attempts": dict(released_counts)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prompt_locks() -> dict[str, Any]:
    """Every frozen judge prompt must hash to the value in prompts/judge/lock.json."""
    from figsbench.judge.stages import JUDGE_DIR, LOCK_PATH

    issues: list[ValidationIssue] = []
    lock = load_json(LOCK_PATH)
    if not isinstance(lock, Mapping) or lock.get("status") != "frozen":
        _issue(issues, "prompts/judge/lock.json.status", "must equal 'frozen'")
        raise SchemaValidationError(issues)
    hashes: dict[str, str] = {}
    for name, stage in lock.get("stages", {}).items():
        path = JUDGE_DIR / stage["prompt"]
        if not path.is_file():
            _issue(issues, f"prompts/judge/{stage['prompt']}", "frozen prompt is missing")
            continue
        hashes[name] = sha256_file(path)
        if hashes[name] != stage["sha256"]:
            _issue(issues, f"prompts/judge/{stage['prompt']}",
                   f"sha256 {hashes[name]} does not match the lock {stage['sha256']}")
    if set(hashes) != {"syc-score", "cv-score", "syc-rules", "cv-rules"}:
        _issue(issues, "prompts/judge/lock.json.stages", "must lock exactly the four stages")
    if issues:
        raise SchemaValidationError(issues)
    return {"locks": len(hashes), "prompts": hashes}


def _find_release_root(candidate: Path) -> Path:
    resolved = candidate.resolve()
    if (resolved / "data/benchmark_500.jsonl").is_file():
        return resolved
    if (resolved.parent / "data/benchmark_500.jsonl").is_file():
        return resolved.parent
    return resolved


def validate_release(root: str | Path) -> dict[str, Any]:
    release_root = _find_release_root(Path(root))
    required = (
        "LICENSE",
        "DATA_LICENSE.md",
        "pyproject.toml",
    )
    missing = [name for name in required if not (release_root / name).is_file()]
    if missing:
        raise SchemaValidationError(
            [ValidationIssue("release", f"required release files are missing: {missing}")]
        )

    samples = load_jsonl(release_root / "data/benchmark_500.jsonl")
    return {
        "release_root": str(release_root),
        "dataset": validate_dataset(samples, canonical=True),
        "gate": validate_gate(
            samples,
            load_json(release_root / "data/provenance/gate_picks.json"),
            load_json(release_root / "data/provenance/gate_manifest.json"),
        ),
        "prompt_locks": validate_prompt_locks(),
        "ok": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_release(args.root)
    except SchemaValidationError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "validation passed: "
            f"{report['dataset']['samples']} samples, {report['gate']['records']} gate records, "
            f"{report['prompt_locks']['locks']} frozen judge prompts"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
