"""Data contracts and semantic validation for source-corpus artifacts."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = 1
PRIMARY_MODEL = "deepseek/deepseek-v4-flash"
ADJUDICATOR_MODEL = "deepseek/deepseek-v4-flash"
SOURCES = {
    "reddit",
    "quora",
    "stackexchange",
    "lemmy",
    "hf_shp",
    "hf_aita",
    "hf_tldr17",
    "hf_reddit_submissions",
    "hf_quora",
    "hf_reddit_finance",
}
SCOPES = {"personal", "general", "unclear"}
TRUTH_TYPES = {"objective", "subjective", "mixed", "undetermined"}
CATEGORY_IDS = {"C1", "C2", "C3", "C4", "C5", "out_of_taxonomy", "unclear"}
ELIGIBILITY = {"eligible", "ineligible", "needs_human_review"}
SAFETY_FLAGS = {
    "minor",
    "self_harm_or_crisis",
    "sexual_content",
    "doxxing",
    "high_risk_medical",
    "high_risk_legal",
    "direct_identifier",
    "none",
}


def _require(mapping: dict[str, Any], fields: set[str], label: str) -> None:
    missing = fields - set(mapping)
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(sorted(missing))}")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _confidence(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} confidence must be numeric")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field} confidence must be between 0 and 1")


def validate_source_record(record: dict[str, Any]) -> None:
    _require(
        record,
        {
            "schema_version",
            "record_id",
            "source",
            "source_item_id",
            "source_url",
            "source_stratum",
            "author_hash",
            "created_at",
            "retrieved_at",
            "title",
            "body",
            "text_sha256",
            "provenance",
        },
        "Source record",
    )
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported source-record schema version")
    if record["source"] not in SOURCES:
        raise ValueError("Unknown source")
    for field in (
        "record_id",
        "source_item_id",
        "source_url",
        "source_stratum",
        "created_at",
        "retrieved_at",
        "text_sha256",
    ):
        if not _nonempty(record[field]):
            raise ValueError(f"Source record {field} must be non-empty text")
    if not isinstance(record["title"], str) or not isinstance(record["body"], str):
        raise ValueError("Source title and body must be strings")
    if not isinstance(record["provenance"], dict):
        raise ValueError("Source provenance must be an object")


def validate_prepared_record(record: dict[str, Any]) -> None:
    _require(
        record,
        {
            "schema_version",
            "record_id",
            "source",
            "source_stratum",
            "created_at",
            "redacted_text",
            "redaction_counts",
            "word_count",
            "prefilter",
            "provenance",
        },
        "Prepared record",
    )
    if record["schema_version"] != SCHEMA_VERSION or record["source"] not in SOURCES:
        raise ValueError("Invalid prepared-record identity")
    eligible = bool(record.get("prefilter", {}).get("eligible"))
    if eligible and not _nonempty(record["redacted_text"]):
        raise ValueError("Eligible prepared record has no redacted text")
    if not isinstance(record["redacted_text"], str):
        raise ValueError("Prepared redacted_text must be text")
    if not isinstance(record["redaction_counts"], dict):
        raise ValueError("redaction_counts must be an object")
    if not isinstance(record["word_count"], int) or record["word_count"] < 0:
        raise ValueError("word_count must be non-negative")
    if not isinstance(record["prefilter"], dict):
        raise ValueError("prefilter must be an object")


def _validate_axis(value: Any, allowed: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    _require(value, {"label", "confidence", "evidence"}, name)
    if value["label"] not in allowed:
        raise ValueError(f"{name} has invalid label {value['label']!r}")
    _confidence(value["confidence"], name)
    if not isinstance(value["evidence"], str):
        raise ValueError(f"{name} evidence must be text")


def validate_model_label(value: dict[str, Any], domains: set[str]) -> None:
    _require(
        value,
        {
            "eligibility",
            "eligibility_reasons",
            "deidentified_summary",
            "domain",
            "domain_scope",
            "ground_truth_type",
            "sycophancy_category",
            "calibrated_validation_category",
            "self_contained",
            "safety_flags",
            "attack_vector_compatibility",
        },
        "Model label",
    )
    _validate_axis(value["domain"], domains | {"out_of_taxonomy", "unclear"}, "domain")
    _validate_axis(value["domain_scope"], SCOPES, "domain_scope")
    _validate_axis(value["ground_truth_type"], TRUTH_TYPES, "ground_truth_type")
    _validate_axis(value["sycophancy_category"], CATEGORY_IDS, "sycophancy_category")
    _validate_axis(
        value["calibrated_validation_category"],
        CATEGORY_IDS,
        "calibrated_validation_category",
    )
    if value["eligibility"] not in ELIGIBILITY:
        raise ValueError("Invalid eligibility")
    if not isinstance(value["eligibility_reasons"], list) or not all(
        isinstance(item, str) and item.strip() for item in value["eligibility_reasons"]
    ):
        raise ValueError("eligibility_reasons must be a non-empty text list")
    if not isinstance(value["deidentified_summary"], str):
        raise ValueError("deidentified_summary must be text")
    if not isinstance(value["self_contained"], bool):
        raise ValueError("self_contained must be boolean")
    if not isinstance(value["safety_flags"], list) or not set(value["safety_flags"]) <= SAFETY_FLAGS:
        raise ValueError("Invalid safety_flags")
    compatibility = value["attack_vector_compatibility"]
    if not isinstance(compatibility, list):
        raise ValueError("attack_vector_compatibility must be a list")
    for item in compatibility:
        if not isinstance(item, dict):
            raise ValueError("Attack compatibility entries must be objects")
        _require(item, {"id", "confidence", "rationale"}, "Attack compatibility")
        if item["id"] not in {"MH", "SR", "AL", "FU", "RL", "PS"}:
            raise ValueError("Unknown attack-vector compatibility ID")
        _confidence(item["confidence"], "attack_vector_compatibility")
        if not isinstance(item["rationale"], str):
            raise ValueError("Attack compatibility rationale must be text")


def validate_label_record(record: dict[str, Any], domains: set[str]) -> None:
    _require(
        record,
        {
            "schema_version",
            "record_id",
            "source",
            "source_stratum",
            "created_at",
            "text_sha256",
            "primary",
            "adjudication",
            "resolved",
            "review_status",
            "model_metadata",
            "provenance",
        },
        "Label record",
    )
    if record["schema_version"] != SCHEMA_VERSION or record["source"] not in SOURCES:
        raise ValueError("Invalid label-record identity")
    validate_model_label(record["primary"], domains)
    if record["adjudication"] is not None:
        validate_model_label(record["adjudication"], domains)
    validate_model_label(record["resolved"], domains)
    if record["review_status"] not in {
        "model_resolved",
        "needs_human_review",
        "human_reviewed",
    }:
        raise ValueError("Invalid review_status")
    if not isinstance(record["model_metadata"], dict):
        raise ValueError("model_metadata must be an object")


def core_confidences(label: dict[str, Any]) -> list[float]:
    return [
        float(label[field]["confidence"])
        for field in (
            "domain",
            "domain_scope",
            "ground_truth_type",
            "sycophancy_category",
            "calibrated_validation_category",
        )
    ]
