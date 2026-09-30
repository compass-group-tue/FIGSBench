"""The FIGS rules (S1.a-S2.d sycophancy, V1-V3 calibrated validation) and the views of them
that authoring prompts and plans use."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from figsbench import DATA_DIR
from figsbench.utils import read_json

RULES_PATH = DATA_DIR / "rules.json"
RULE_SNAPSHOT_VERSION = "figs-rules-v1"
EXPECTED_ATOMIC_RULE_COUNTS = {"sycophancy": 7, "calibrated_validation": 3}


def load_rule_snapshot(path: Path = RULES_PATH) -> dict[str, Any]:
    value = read_json(path)
    required = {
        "schema_version",
        "rule_snapshot_version",
        "source",
        "constructs",
        "s_system",
        "validation_system",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("rules file has the wrong top-level fields")
    return value


@lru_cache(maxsize=1)
def cached_rule_snapshot() -> dict[str, Any]:
    value = read_json(RULES_PATH)
    if value.get("rule_snapshot_version") != RULE_SNAPSHOT_VERSION:
        raise ValueError("rules file version mismatch")
    return value


def _rule_view(rule: Mapping[str, Any], axis: str) -> dict[str, Any]:
    return {
        "rule_id": rule["id"],
        "chunk_id": str(rule["id"]),
        "title": rule["title"],
        "axis": axis,
        "primary_rule_ids": [rule["id"]],
        "description": rule["description"],
        "example": rule["example"],
        "neighboring_boundary": rule["neighboring_boundary"],
    }


def atomic_rules_by_axis(
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    snapshot = snapshot or load_rule_snapshot()
    rules = {
        "sycophancy": [
            _rule_view(rule, "sycophancy")
            for rule in [
                *snapshot["s_system"]["s1_conditions"],
                *snapshot["s_system"]["s2_loci"],
            ]
        ],
        "calibrated_validation": [
            _rule_view(rule, "calibrated_validation")
            for rule in snapshot["validation_system"]["rules"]
        ],
    }
    counts = {axis: len(values) for axis, values in rules.items()}
    if counts != EXPECTED_ATOMIC_RULE_COUNTS:
        raise ValueError(f"unexpected rule counts: {counts}")
    return rules


def selected_validation_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluated_rule record of a calibrated-validation assignment."""
    return {
        "chunk_id": rule["chunk_id"],
        "title": rule["title"],
        "axis": "calibrated_validation",
        "primary_rule_ids": [rule["rule_id"]],
        "rule": deepcopy(rule),
    }


def _rule_index(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rules = [
        *snapshot["s_system"]["s1_conditions"],
        *snapshot["s_system"]["s2_loci"],
        *snapshot["validation_system"]["rules"],
    ]
    return {str(rule["id"]): rule for rule in rules}


def _collect_selected_rule_ids(value: Any, known: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        if value in known:
            found.add(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key in {
                "id",
                "rule_id",
                "s1",
                "s2",
                "validation",
                "rule",
            }:
                found.update(_collect_selected_rule_ids(item, known))
            elif key in {"primary_rule_ids", "rule_ids", "selected_rule_ids"}:
                found.update(_collect_selected_rule_ids(item, known))
            elif isinstance(item, Mapping):
                found.update(_collect_selected_rule_ids(item, known))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.update(_collect_selected_rule_ids(item, known))
    return found


def selected_rule_context(assignment: Mapping[str, Any]) -> dict[str, Any]:
    """The one target rule of an assignment, with its canonical definition."""
    snapshot = cached_rule_snapshot()
    index = _rule_index(snapshot)
    selection = assignment.get("selected_rules", assignment.get("evaluated_rule"))
    selected_ids = _collect_selected_rule_ids(selection, set(index))
    axis = assignment["benchmark_axis"]
    if axis == "sycophancy":
        s_rules = sorted(rule_id for rule_id in selected_ids if rule_id.startswith("S"))
        if len(s_rules) != 1:
            raise ValueError("sycophancy assignment must select one atomic S rule")
        ordered_ids = s_rules
    elif axis == "calibrated_validation":
        validation = sorted(
            rule_id for rule_id in selected_ids if rule_id.startswith("V")
        )
        if len(validation) != 1:
            raise ValueError(
                "calibrated-validation assignment must select one canonical V rule"
            )
        ordered_ids = validation
    else:
        raise ValueError(f"unsupported benchmark axis: {axis}")
    return {
        "selected_rule_ids": ordered_ids,
        "canonical_definitions": [deepcopy(index[rule_id]) for rule_id in ordered_ids],
    }


def complete_rule_context() -> dict[str, Any]:
    snapshot = cached_rule_snapshot()
    return {
        "s_system": deepcopy(snapshot["s_system"]),
        "validation_system": deepcopy(snapshot["validation_system"]),
    }


def complete_sycophancy_rule_context() -> dict[str, Any]:
    snapshot = cached_rule_snapshot()
    return {"s_system": deepcopy(snapshot["s_system"])}


def axis_motivation(axis: str) -> str:
    snapshot = cached_rule_snapshot()
    if axis == "sycophancy":
        return snapshot["constructs"]["sycophancy"]
    if axis == "calibrated_validation":
        return snapshot["constructs"]["calibrated_validation"]
    raise ValueError(f"unsupported benchmark axis: {axis}")
