"""Deterministic difficulty-gate selection and benchmark release assembly.

The builder is deliberately network-free.  It combines two independently
authored attempts, reads their already-computed judge results, selects the
attempt with the larger (worse) score on each slot's target axis, and writes
the provenance needed to audit that choice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AXES = ("sycophancy", "calibrated_validation")
_AXIS_BY_PREFIX = {"SYC": "sycophancy", "CV": "calibrated_validation"}
_SLOT_ID_RE = re.compile(r"^(?:FINAL-)?(SYC|CV)-(\d{3})(?:-X4)?$")
_SLOT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9-])(?:FINAL-)?(SYC|CV)-(\d{3})(?:-X4)?(?![A-Za-z0-9-])"
)


class ReleaseBuildError(ValueError):
    """Raised when source artifacts cannot form an unambiguous release."""


@dataclass(frozen=True)
class AssembledRelease:
    """In-memory, deterministic release artifacts."""

    records: list[dict[str, Any]]
    gate_picks: dict[str, dict[str, Any]]
    gate_manifest: dict[str, Any]


@dataclass(frozen=True)
class ReleaseArtifacts:
    """Paths written by :func:`build_release`."""

    gate_picks: Path
    gate_manifest: Path
    unexpanded_jsonl: Path
    benchmark_jsonl: Path | None


def canonical_slot_id(value: Any) -> str:
    """Return the canonical ``FINAL-{SYC,CV}-NNN`` form of a slot ID."""

    if not isinstance(value, str):
        raise ReleaseBuildError(f"slot id must be a string, got {value!r}")
    match = _SLOT_ID_RE.fullmatch(value.strip())
    if match is None:
        raise ReleaseBuildError(
            f"invalid slot id {value!r}; expected FINAL-SYC-NNN or FINAL-CV-NNN"
        )
    return f"FINAL-{match.group(1)}-{match.group(2)}"


def canonicalize_ids(value: Any) -> Any:
    """Recursively canonicalize slot-ID tokens, including stale ``-X4`` IDs.

    Dictionary keys are canonicalized as well as values.  A collision caused
    by canonicalization is rejected instead of silently dropping data.
    """

    if isinstance(value, str):
        return _SLOT_TOKEN_RE.sub(
            lambda match: f"FINAL-{match.group(1)}-{match.group(2)}", value
        )
    if isinstance(value, list):
        return [canonicalize_ids(item) for item in value]
    if isinstance(value, tuple):
        return tuple(canonicalize_ids(item) for item in value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            canonical_key = canonicalize_ids(key)
            if canonical_key in out:
                raise ReleaseBuildError(
                    f"ID canonicalization caused duplicate key {canonical_key!r}"
                )
            out[canonical_key] = canonicalize_ids(item)
        return out
    return value


def slot_sort_key(slot_id: str) -> tuple[int, int]:
    """Sort CV slots first, matching the published release, then by ordinal."""

    canonical = canonical_slot_id(slot_id)
    match = _SLOT_ID_RE.fullmatch(canonical)
    assert match is not None
    return (0 if match.group(1) == "CV" else 1, int(match.group(2)))


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSONL file and report line-local parse/type errors."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseBuildError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ReleaseBuildError(
                    f"{source}:{line_number}: expected a JSON object"
                )
            records.append(record)
    if not records:
        raise ReleaseBuildError(f"{source}: no JSONL records found")
    return records


def _samples_file(run: str | os.PathLike[str]) -> Path:
    path = Path(run)
    if path.is_file():
        return path
    candidate = path / "samples.jsonl"
    if not candidate.is_file():
        raise FileNotFoundError(f"sample run has no samples.jsonl: {path}")
    return candidate


def load_samples(run_dirs: Sequence[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """Load and concatenate ``samples.jsonl`` from one or more run directories."""

    if not run_dirs:
        raise ReleaseBuildError("at least one sample run is required")
    records: list[dict[str, Any]] = []
    for run in run_dirs:
        records.extend(read_jsonl(_samples_file(run)))
    return records


def _judge_files(run: str | os.PathLike[str]) -> list[Path]:
    path = Path(run)
    if path.is_file():
        return [path]
    results = path / "results" if (path / "results").is_dir() else path
    files = sorted(results.glob("*.json"))
    if not files:
        jsonl = sorted(results.glob("*.jsonl"))
        if jsonl:
            return jsonl
        raise FileNotFoundError(f"judge run has no result JSON files: {path}")
    return files


def load_judge_results(
    run_dirs: Sequence[str | os.PathLike[str]],
) -> list[dict[str, Any]]:
    """Load per-sample results from one or more judge run directories.

    A run can be a directory containing ``results/*.json``, the ``results``
    directory itself, a JSONL file, or a single result JSON file.
    """

    if not run_dirs:
        raise ReleaseBuildError("at least one judge run is required")
    records: list[dict[str, Any]] = []
    for run in run_dirs:
        for path in _judge_files(run):
            if path.suffix == ".jsonl":
                records.extend(read_jsonl(path))
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ReleaseBuildError(f"{path}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ReleaseBuildError(f"{path}: expected a JSON object")
            records.append(value)
    return records


def _sample_identity(sample: Mapping[str, Any], *, source: str) -> tuple[str, str]:
    if "id" not in sample:
        raise ReleaseBuildError(f"{source}: sample is missing id")
    slot_id = canonical_slot_id(sample["id"])
    candidate_ids: list[tuple[str, Any]] = []
    if sample.get("final_slot_id") is not None:
        candidate_ids.append(("final_slot_id", sample["final_slot_id"]))
    scenario = sample.get("scenario")
    if isinstance(scenario, Mapping) and scenario.get("id") is not None:
        candidate_ids.append(("scenario.id", scenario["id"]))
    for field, value in candidate_ids:
        try:
            candidate = canonical_slot_id(value)
        except ReleaseBuildError:
            # Some historical scenario IDs were not slot IDs.  Only validate
            # values that claim the FINAL-SYC/CV namespace.
            if isinstance(value, str) and ("SYC-" in value or "CV-" in value):
                raise ReleaseBuildError(f"{source} {slot_id}: invalid {field}={value!r}")
            continue
        if candidate != slot_id:
            raise ReleaseBuildError(
                f"{source} {slot_id}: {field} points to {candidate}"
            )

    declared_axes = [sample.get("benchmark_axis")]
    if isinstance(scenario, Mapping):
        declared_axes.append(scenario.get("benchmark_axis"))
    axes = {axis for axis in declared_axes if axis is not None}
    if len(axes) != 1:
        raise ReleaseBuildError(
            f"{source} {slot_id}: expected one consistent benchmark_axis, got {sorted(axes)!r}"
        )
    axis = next(iter(axes))
    if axis not in AXES:
        raise ReleaseBuildError(f"{source} {slot_id}: unknown benchmark_axis {axis!r}")
    prefix = slot_id.split("-")[1]
    expected = _AXIS_BY_PREFIX[prefix]
    if axis != expected:
        raise ReleaseBuildError(
            f"{source} {slot_id}: id implies {expected}, sample declares {axis}"
        )
    return slot_id, axis


def _index_samples(
    samples: Iterable[Mapping[str, Any]], *, source: str
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    indexed: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ReleaseBuildError(f"{source}: sample is not an object")
        slot_id, axis = _sample_identity(sample, source=source)
        if slot_id in indexed:
            raise ReleaseBuildError(f"{source}: duplicate slot {slot_id}")
        indexed[slot_id] = (axis, sample)
    if not indexed:
        raise ReleaseBuildError(f"{source}: no samples")
    return indexed


def _result_id(result: Mapping[str, Any], *, source: str) -> str:
    raw = result.get("sample_id", result.get("id"))
    if raw is None:
        raise ReleaseBuildError(f"{source}: judge result is missing sample_id")
    return canonical_slot_id(raw)


def _index_results(
    results: Iterable[Mapping[str, Any]], *, source: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise ReleaseBuildError(f"{source}: judge result is not an object")
        slot_id = _result_id(result, source=source)
        if slot_id in indexed:
            raise ReleaseBuildError(f"{source}: duplicate judge result for {slot_id}")
        indexed[slot_id] = result
    if not indexed:
        raise ReleaseBuildError(f"{source}: no judge results")
    return indexed


def _require_matching_ids(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_name: str,
    right_name: str,
) -> None:
    missing = sorted(set(left) - set(right), key=slot_sort_key)
    extra = sorted(set(right) - set(left), key=slot_sort_key)
    if missing or extra:
        raise ReleaseBuildError(
            f"{left_name}/{right_name} slot mismatch: missing={missing}, extra={extra}"
        )


def _target_score(result: Mapping[str, Any], axis: str, *, source: str) -> int:
    status = result.get("status")
    if status is not None and status != "success":
        raise ReleaseBuildError(f"{source}: judge status is {status!r}, not 'success'")

    judgment = result.get("judgment", result)
    if not isinstance(judgment, Mapping):
        raise ReleaseBuildError(f"{source}: judgment is not an object")
    axis_scored = judgment.get("axis_scored", result.get("axis_scored"))
    if axis_scored not in (None, "both", axis):
        raise ReleaseBuildError(
            f"{source}: result scored {axis_scored!r}, expected {axis!r} or 'both'"
        )
    declared_axis = result.get("benchmark_axis")
    if declared_axis is not None and declared_axis != axis:
        raise ReleaseBuildError(
            f"{source}: result declares {declared_axis!r}, expected {axis!r}"
        )

    axis_value = judgment.get(axis)
    if isinstance(axis_value, Mapping):
        score = axis_value.get("score")
    else:
        score = judgment.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 4:
        raise ReleaseBuildError(f"{source}: target-axis score must be an integer 1-4")
    return score


def _normalize_stored_picks(
    stored_picks: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]] | None:
    if stored_picks is None:
        return None
    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_id, pick in stored_picks.items():
        slot_id = canonical_slot_id(raw_id)
        if slot_id in normalized:
            raise ReleaseBuildError(f"stored picks contain duplicate slot {slot_id}")
        if not isinstance(pick, Mapping):
            raise ReleaseBuildError(f"stored pick {slot_id} is not an object")
        normalized[slot_id] = pick
    return normalized


def assemble_release(
    *,
    attempt_a_samples: Iterable[Mapping[str, Any]],
    attempt_b_samples: Iterable[Mapping[str, Any]],
    attempt_a_results: Iterable[Mapping[str, Any]],
    attempt_b_results: Iterable[Mapping[str, Any]],
    gate_model: str,
    tie_winner: str = "a",
    stored_picks: Mapping[str, Any] | None = None,
    benchmark_name: str | None = None,
) -> AssembledRelease:
    """Validate, gate, and canonicalize a release entirely in memory."""

    if tie_winner not in ("a", "b"):
        raise ReleaseBuildError("tie_winner must be 'a' or 'b'")
    if not isinstance(gate_model, str) or not gate_model.strip():
        raise ReleaseBuildError("gate_model must be a non-empty string")

    samples_a = _index_samples(attempt_a_samples, source="attempt a")
    samples_b = _index_samples(attempt_b_samples, source="attempt b")
    _require_matching_ids(
        samples_a, samples_b, left_name="attempt a", right_name="attempt b"
    )
    results_a = _index_results(attempt_a_results, source="attempt a judge")
    results_b = _index_results(attempt_b_results, source="attempt b judge")
    _require_matching_ids(
        samples_a, results_a, left_name="attempt a", right_name="attempt a judge"
    )
    _require_matching_ids(
        samples_b, results_b, left_name="attempt b", right_name="attempt b judge"
    )
    saved = _normalize_stored_picks(stored_picks)
    if saved is not None:
        _require_matching_ids(
            samples_a, saved, left_name="samples", right_name="stored picks"
        )

    records: list[dict[str, Any]] = []
    picks: dict[str, dict[str, Any]] = {}
    for slot_id in sorted(samples_a, key=slot_sort_key):
        axis_a, sample_a = samples_a[slot_id]
        axis_b, sample_b = samples_b[slot_id]
        if axis_a != axis_b:
            raise ReleaseBuildError(
                f"{slot_id}: attempt axis mismatch ({axis_a!r} vs {axis_b!r})"
            )
        score_a = _target_score(
            results_a[slot_id], axis_a, source=f"attempt a judge {slot_id}"
        )
        score_b = _target_score(
            results_b[slot_id], axis_a, source=f"attempt b judge {slot_id}"
        )
        tied = score_a == score_b
        harder = "a" if score_a > score_b else "b"
        released = tie_winner if tied else harder

        if saved is not None:
            stored = saved[slot_id]
            expected_fields = {
                "score_a": score_a,
                "score_b": score_b,
                "tied": tied,
            }
            for field, expected in expected_fields.items():
                if stored.get(field) != expected:
                    raise ReleaseBuildError(
                        f"stored pick {slot_id}: {field}={stored.get(field)!r}, "
                        f"expected {expected!r}"
                    )
            stored_attempt = stored.get("released_attempt")
            if stored_attempt not in ("a", "b"):
                raise ReleaseBuildError(
                    f"stored pick {slot_id}: released_attempt must be 'a' or 'b'"
                )
            if not tied and stored_attempt != harder:
                raise ReleaseBuildError(
                    f"stored pick {slot_id}: cannot override strictly harder attempt {harder}"
                )
            if tied:
                released = stored_attempt

        pick = {
            "score_a": score_a,
            "score_b": score_b,
            "tied": tied,
            "released_attempt": released,
        }
        picks[slot_id] = pick
        selected = sample_a if released == "a" else sample_b
        record = canonicalize_ids(deepcopy(dict(selected)))
        record["id"] = slot_id
        record["gate_winner_attempt"] = released
        record["gate_filter"] = gate_model
        records.append(record)

    axis_counts = Counter(record["benchmark_axis"] for record in records)
    attempt_counts = Counter(pick["released_attempt"] for pick in picks.values())
    tied_count = sum(bool(pick["tied"]) for pick in picks.values())
    count = len(records)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": benchmark_name or f"final-{count}",
        "gate_model": gate_model,
        "gate": (
            "Each slot was authored twice (attempts a and b) and judged on its "
            "own benchmark axis. The released attempt is the one with the larger "
            "(worse) target-axis score; ties use the recorded deterministic tie policy."
        ),
        "n": count,
        "axes": {axis: axis_counts.get(axis, 0) for axis in AXES},
        "tied": tied_count,
        "released_attempt_counts": {
            attempt: attempt_counts.get(attempt, 0) for attempt in ("a", "b")
        },
        "tie_policy": {
            "default_winner": tie_winner,
            "stored_picks_applied": saved is not None,
        },
    }
    return AssembledRelease(records=records, gate_picks=picks, gate_manifest=manifest)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprints(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            # Keep machine-specific absolute paths out of a public manifest.
            "path": f"{path.parent.name}/{path.name}",
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _judge_source_fingerprints(
    runs: Sequence[str | os.PathLike[str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in runs:
        paths = _judge_files(run)
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_sha256_file(path)))
        out.append(
            {
                "path": Path(run).name,
                "files": len(paths),
                "sha256": digest.hexdigest(),
            }
        )
    return out


def _expanded_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    projected = deepcopy(dict(record))
    scenario = projected.get("scenario")
    if not isinstance(scenario, dict):
        raise ReleaseBuildError(f"{projected.get('id')}: scenario must be an object")
    scenario.pop("scenario_plan", None)
    scenario.pop("expansion", None)
    scenario.pop("original_scenario_plan", None)
    return projected


def validate_expanded_records(
    unexpanded: Sequence[Mapping[str, Any]],
    expanded: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate a pre-expanded file one-to-one and return canonical slot order."""

    source = _index_samples(unexpanded, source="unexpanded release")
    candidate = _index_samples(expanded, source="expanded input")
    _require_matching_ids(
        source, candidate, left_name="unexpanded release", right_name="expanded input"
    )
    output: list[dict[str, Any]] = []
    for slot_id in sorted(source, key=slot_sort_key):
        _, base_raw = source[slot_id]
        _, expanded_raw = candidate[slot_id]
        base = canonicalize_ids(deepcopy(dict(base_raw)))
        item = canonicalize_ids(deepcopy(dict(expanded_raw)))
        if _expanded_projection(base) != _expanded_projection(item):
            raise ReleaseBuildError(
                f"expanded input {slot_id}: fields other than expansion metadata/scenario_plan changed"
            )
        base_scenario = base.get("scenario")
        item_scenario = item.get("scenario")
        assert isinstance(base_scenario, dict) and isinstance(item_scenario, dict)
        plan = item_scenario.get("scenario_plan")
        if not isinstance(plan, str) or not plan.strip():
            raise ReleaseBuildError(
                f"expanded input {slot_id}: scenario.scenario_plan must be non-empty"
            )
        original_plan = item_scenario.get("original_scenario_plan")
        if original_plan is not None and original_plan != base_scenario.get("scenario_plan"):
            raise ReleaseBuildError(
                f"expanded input {slot_id}: original_scenario_plan does not match the source"
            )
        item["id"] = slot_id
        output.append(item)
    return output


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Linking a same-directory temporary into place is atomic and,
            # unlike os.replace(), fails if a concurrent writer created the
            # destination after our overwrite preflight.
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_release(
    *,
    attempt_a_runs: Sequence[str | os.PathLike[str]],
    attempt_b_runs: Sequence[str | os.PathLike[str]],
    attempt_a_judge_runs: Sequence[str | os.PathLike[str]],
    attempt_b_judge_runs: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    gate_model: str,
    tie_winner: str = "a",
    stored_picks_path: str | os.PathLike[str] | None = None,
    expanded_jsonl: str | os.PathLike[str] | None = None,
    benchmark_name: str | None = None,
    force: bool = False,
) -> ReleaseArtifacts:
    """Build and atomically write a complete, offline release bundle.

    ``attempt_*_runs`` may each contain one combined run or multiple run
    directories (for example separate sycophancy and CV ``run_final`` runs).
    Existing output files are never replaced unless ``force`` is true.
    """

    sample_files_a = [_samples_file(run) for run in attempt_a_runs]
    sample_files_b = [_samples_file(run) for run in attempt_b_runs]
    samples_a = load_samples(attempt_a_runs)
    samples_b = load_samples(attempt_b_runs)
    results_a = load_judge_results(attempt_a_judge_runs)
    results_b = load_judge_results(attempt_b_judge_runs)

    stored_picks: Mapping[str, Any] | None = None
    stored_path: Path | None = None
    if stored_picks_path is not None:
        stored_path = Path(stored_picks_path)
        stored_value = json.loads(stored_path.read_text(encoding="utf-8"))
        if not isinstance(stored_value, dict):
            raise ReleaseBuildError("stored picks file must contain a JSON object")
        stored_picks = stored_value

    assembled = assemble_release(
        attempt_a_samples=samples_a,
        attempt_b_samples=samples_b,
        attempt_a_results=results_a,
        attempt_b_results=results_b,
        gate_model=gate_model,
        tie_winner=tie_winner,
        stored_picks=stored_picks,
        benchmark_name=benchmark_name,
    )

    output_root = Path(output_dir)
    picks_path = output_root / "gate_picks.json"
    manifest_path = output_root / "gate_manifest.json"
    unexpanded_path = output_root / "benchmark_unexpanded.jsonl"
    benchmark_path = output_root / "benchmark.jsonl" if expanded_jsonl else None
    destinations = [picks_path, manifest_path, unexpanded_path]

    expanded_records: list[dict[str, Any]] | None = None
    expanded_path: Path | None = None
    if expanded_jsonl is not None:
        expanded_path = Path(expanded_jsonl)
        expanded_records = validate_expanded_records(
            assembled.records, read_jsonl(expanded_path)
        )
        assert benchmark_path is not None
        destinations.append(benchmark_path)

    existing = [path for path in destinations if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing release files without force: "
            + ", ".join(path.as_posix() for path in existing)
        )

    picks_payload = _json_bytes(assembled.gate_picks)
    unexpanded_payload = _jsonl_bytes(assembled.records)
    benchmark_payload = (
        _jsonl_bytes(expanded_records) if expanded_records is not None else None
    )
    manifest = deepcopy(assembled.gate_manifest)
    manifest["inputs"] = {
        "attempt_a_samples": _source_fingerprints(sample_files_a),
        "attempt_b_samples": _source_fingerprints(sample_files_b),
        "attempt_a_judgments": _judge_source_fingerprints(attempt_a_judge_runs),
        "attempt_b_judgments": _judge_source_fingerprints(attempt_b_judge_runs),
        "stored_picks": (
            {
                "path": stored_path.name,
                "sha256": _sha256_file(stored_path),
            }
            if stored_path is not None
            else None
        ),
        "expanded_jsonl": (
            {
                "path": expanded_path.name,
                "sha256": _sha256_file(expanded_path),
            }
            if expanded_path is not None
            else None
        ),
    }
    manifest["outputs"] = {
        "gate_picks": {
            "path": picks_path.name,
            "sha256": _sha256_bytes(picks_payload),
        },
        "unexpanded_jsonl": {
            "path": unexpanded_path.name,
            "sha256": _sha256_bytes(unexpanded_payload),
        },
        "benchmark_jsonl": (
            {
                "path": benchmark_path.name,
                "sha256": _sha256_bytes(benchmark_payload),
            }
            if benchmark_path is not None and benchmark_payload is not None
            else None
        ),
    }
    manifest_payload = _json_bytes(manifest)

    # Validation and the overwrite preflight happen before the first write.
    _atomic_write(picks_path, picks_payload, overwrite=force)
    _atomic_write(unexpanded_path, unexpanded_payload, overwrite=force)
    if benchmark_path is not None and benchmark_payload is not None:
        _atomic_write(benchmark_path, benchmark_payload, overwrite=force)
    _atomic_write(manifest_path, manifest_payload, overwrite=force)
    return ReleaseArtifacts(
        gate_picks=picks_path,
        gate_manifest=manifest_path,
        unexpanded_jsonl=unexpanded_path,
        benchmark_jsonl=benchmark_path,
    )
