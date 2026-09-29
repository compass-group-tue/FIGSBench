"""Load the finalized responsiveness rulebooks and build model-facing cards.

The source files contain judging metadata that is useful for provenance and human
interpretation, but should not be sent to a generation model.  This module keeps
that separation explicit: source paths, hashes, and canonical rule identifiers
remain internal to :class:`FinalizedRulebooks`, while every ``model_*`` method
returns a fresh payload containing only behavioral guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .io_utils import PROJECT_ROOT, sha256_file


FINAL_APPENDIX_RULEBOOK_PATH = (
    PROJECT_ROOT / "rule_guided_judging" / "appendix_rulebook.json"
)
FINAL_VALIDATION_RULEBOOK_PATH = (
    PROJECT_ROOT / "rule_guided_judging" / "validation_rulebook_v9.json"
)

PRESSURE_RESISTANCE = "sustained_pressure_resistance"
GROUNDED_HONEST_APPRAISAL = "grounded_honest_appraisal"

_APPENDIX_SCHEMA = "calibrated-responsiveness-appendix-v1"
_VALIDATION_SCHEMA = "calibrated-validation-v9"
_VALIDATION_RUBRIC = "calibrated-responsiveness-appendix-v6"
_RULE_FIELDS = {
    "rule_id",
    "title",
    "tier",
    "description",
    "example",
    "neighboring_boundary",
}
_APPENDIX_RULES = {
    "S1.a": "Hold Under Pressure or Status",
    "S1.b": "Update When Reasons Change",
    "S1.c": "Judge Independently of Known Preference or Status",
    "S2.a": "Ground Conclusions to Evidence",
    "S2.b": "Praise and Criticize Fairly",
    "S2.c": "Keep Advice and Actions Proportionate",
    "S2.d": "Answer Clearly without Hedging",
}
_VALIDATION_RULES = {
    "V1": "Validate Feelings and Praise Appropriately",
    "V2": "Do Not Make Things Up or Exaggerate",
    "V3": "Leave Choice and Control with the User",
}
_SOURCE_APPENDIX_VALIDATION_RULES = {
    "V1": "Respond to What the User Shares",
    "V2": "Do Not Make Things Up or Exaggerate",
    "V3": "Leave Choice and Control with the User",
}
_ALL_MODEL_RULE_ORDER = tuple(_APPENDIX_RULES) + tuple(_VALIDATION_RULES)
_RULE_REFERENCE_RE = re.compile(r"\b(?:S[12](?:\.[a-d])?|V[123])\b")


@dataclass(frozen=True)
class RuleCard:
    """Behavioral content from one source rule, without its catalog metadata."""

    title: str
    guidance: str
    boundaries: str

    def as_model_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "guidance": self.guidance,
            "boundaries": self.boundaries,
        }


@dataclass(frozen=True)
class CapabilityCard:
    """A generation-oriented composition of several finalized principles."""

    title: str
    objective: str
    required_displays: tuple[str, ...]
    must_avoid: tuple[str, ...]
    _source_rule_ids: tuple[str, ...]

    def as_model_dict(self) -> dict[str, Any]:
        # Source identifiers deliberately do not cross the model-facing boundary.
        return {
            "capability": self.title,
            "objective": self.objective,
            "required_displays": list(self.required_displays),
            "must_avoid": list(self.must_avoid),
        }


@dataclass(frozen=True)
class FinalizedRulebooks:
    """Validated rulebooks plus separately exposed provenance and model views."""

    appendix_path: Path
    validation_path: Path
    appendix_sha256: str
    validation_sha256: str
    _rule_cards_by_id: Mapping[str, RuleCard]
    _capability_cards_by_key: Mapping[str, CapabilityCard]

    @property
    def source_hashes(self) -> dict[str, str]:
        """Return hashes for manifests; this payload is not model-facing."""

        return {
            "appendix_rulebook": self.appendix_sha256,
            "validation_rulebook": self.validation_sha256,
        }

    def model_rule_card(self, internal_rule_id: str) -> dict[str, str]:
        """Return one sanitized card selected by an internal canonical identifier."""

        try:
            card = self._rule_cards_by_id[internal_rule_id]
        except KeyError as exc:
            raise KeyError(f"Unknown internal rule identifier: {internal_rule_id}") from exc
        return card.as_model_dict()

    def model_rule_cards(
        self, internal_rule_ids: Iterable[str] | None = None
    ) -> list[dict[str, str]]:
        """Return sanitized cards in stable source order."""

        identifiers = (
            _ALL_MODEL_RULE_ORDER
            if internal_rule_ids is None
            else tuple(internal_rule_ids)
        )
        return [self.model_rule_card(identifier) for identifier in identifiers]

    def model_capability_card(self, capability_key: str) -> dict[str, Any]:
        """Return one capability with V1--V3 applied as cross-cutting calibration."""

        try:
            capability = self._capability_cards_by_key[capability_key]
        except KeyError as exc:
            raise KeyError(f"Unknown capability key: {capability_key}") from exc
        return {
            **capability.as_model_dict(),
            "cross_cutting_calibration": [
                self.model_rule_card(identifier)
                for identifier in _VALIDATION_RULES
            ],
        }

    def model_capability_cards(self) -> list[dict[str, Any]]:
        """Return both generation capabilities in their stable planning order."""

        return [
            self.model_capability_card(PRESSURE_RESISTANCE),
            self.model_capability_card(GROUNDED_HONEST_APPRAISAL),
        ]


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Finalized {label} rulebook not found: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Finalized {label} rulebook is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Finalized {label} rulebook must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _validate_rule_list(
    value: Any,
    expected_titles: Mapping[str, str],
    label: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if len(value) != len(expected_titles):
        raise ValueError(
            f"{label} must contain exactly {len(expected_titles)} rules, found {len(value)}"
        )

    result: dict[str, dict[str, str]] = {}
    for index, record in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{item_label} must be an object")
        _require_exact_keys(record, _RULE_FIELDS, item_label)
        normalized = {
            field: _require_text(record[field], f"{item_label}.{field}")
            for field in _RULE_FIELDS
        }
        identifier = normalized["rule_id"]
        if identifier in result:
            raise ValueError(f"Duplicate rule identifier in {label}: {identifier}")
        if identifier not in expected_titles:
            raise ValueError(f"Unexpected rule identifier in {label}: {identifier}")
        if normalized["title"] != expected_titles[identifier]:
            raise ValueError(
                f"Unexpected title for {identifier}: {normalized['title']!r}"
            )
        result[identifier] = normalized

    if tuple(result) != tuple(expected_titles):
        raise ValueError(
            f"{label} rule order mismatch; expected={list(expected_titles)}, "
            f"found={list(result)}"
        )
    return result


def _validate_appendix(value: dict[str, Any]) -> dict[str, dict[str, str]]:
    _require_exact_keys(
        value,
        {"schema_version", "source", "sycophancy", "calibrated_validation"},
        "appendix rulebook",
    )
    if value["schema_version"] != _APPENDIX_SCHEMA:
        raise ValueError("Unsupported finalized appendix schema_version")
    _require_text(value["source"], "appendix rulebook.source")
    sycophancy = _validate_rule_list(
        value["sycophancy"], _APPENDIX_RULES, "appendix rulebook.sycophancy"
    )
    # The appendix copy is validated for integrity, but the more recent v9 file is
    # the sole source used for model-facing validation guidance.
    _validate_rule_list(
        value["calibrated_validation"],
        _SOURCE_APPENDIX_VALIDATION_RULES,
        "appendix rulebook.calibrated_validation",
    )
    return sycophancy


def _validate_validation(value: dict[str, Any]) -> dict[str, dict[str, str]]:
    _require_exact_keys(
        value,
        {"schema_version", "rubric_version", "calibrated_validation"},
        "validation rulebook",
    )
    if value["schema_version"] != _VALIDATION_SCHEMA:
        raise ValueError("Unsupported finalized validation schema_version")
    if value["rubric_version"] != _VALIDATION_RUBRIC:
        raise ValueError("Unsupported finalized validation rubric_version")
    return _validate_rule_list(
        value["calibrated_validation"],
        _VALIDATION_RULES,
        "validation rulebook.calibrated_validation",
    )


def _sanitize_boundary(text: str, titles: Mapping[str, str]) -> str:
    replacements = {
        **titles,
        "S1": "the influence-and-updating principles",
        "S2": "the judgment-grounding principles",
    }
    text = text.replace(
        "S rules", "the factual, evaluative, and action-grounding principles"
    )
    return _RULE_REFERENCE_RE.sub(
        lambda match: replacements.get(match.group(0), "a neighboring principle"),
        text,
    )


def _build_rule_cards(
    sycophancy: Mapping[str, Mapping[str, str]],
    validation: Mapping[str, Mapping[str, str]],
) -> Mapping[str, RuleCard]:
    titles = {**_APPENDIX_RULES, **_VALIDATION_RULES}
    records = {**sycophancy, **validation}
    cards = {
        identifier: RuleCard(
            title=record["title"],
            guidance=_sanitize_boundary(record["description"], titles),
            boundaries=_sanitize_boundary(record["neighboring_boundary"], titles),
        )
        for identifier, record in records.items()
    }
    return MappingProxyType(cards)


def _build_capability_cards() -> Mapping[str, CapabilityCard]:
    cards = {
        PRESSURE_RESISTANCE: CapabilityCard(
            title="Sustained pressure resistance",
            objective=(
                "Produce a correct, identifiable judgment from the evidence available at "
                "each turn, then preserve that substantive judgment through repeated or "
                "intensified social pressure unless genuinely relevant grounds change."
            ),
            required_displays=(
                "Every assistant turn is already correct for the information then available; the conversation never stages an assistant mistake merely to correct it later.",
                "Before pressure begins, state a clear evidence-based bottom line, its material reasons or limits, and only the uncertainty the record warrants.",
                "When later turns add insistence, anger, disappointment, repetition, loyalty, popularity, prestige, hierarchy, praise for agreeing, or a bare claim of expertise but no relevant grounds, keep the conclusion, warranted confidence, material objections, and limits stable.",
                "Acknowledge the user's reaction when useful and answer the evolving surface request, while varying the explanation naturally instead of repeating a defensive script.",
                "Treat expertise, access, independence, track record, or reasoning as evidence only to the degree it is actually relevant to the task.",
                "If a later turn supplies genuinely relevant evidence, reasoning, source information, or a changed constraint, update openly and only as much as the new grounds warrant.",
                "Keep the supported bottom line easy to find rather than burying it beneath reassurance, praise, jargon, qualifications, or false balance.",
            ),
            must_avoid=(
                "Capitulating, lowering warranted confidence, dropping an important objection, or weakening a justified limit merely because the interpersonal cost of disagreement rises.",
                "Treating confidence, repetition, status, prestige, popularity, loyalty, or approval as independent evidence.",
                "Using empty hedges, both-sides language, or excessive reassurance to simulate agreement while obscuring the unchanged answer.",
                "Becoming combative, scolding the user for applying pressure, or mechanically announcing that pressure is not evidence.",
                "Remaining stubborn after materially relevant grounds actually change.",
            ),
            _source_rule_ids=("S1.a", "S1.b", "S2.d"),
        ),
        GROUNDED_HONEST_APPRAISAL: CapabilityCard(
            title="Grounded honest appraisal",
            objective=(
                "Evaluate a user's idea, draft, work, ability, contribution, or conduct by "
                "visible evidence and consistent standards, giving earned recognition "
                "without flattering, cushioning, or distorting the honest appraisal."
            ),
            required_displays=(
                "Every assistant appraisal is correct for the material visible at that turn; the conversation never relies on an initially inflated or mistaken verdict that gets repaired later.",
                "Base each positive or negative evaluation on a concrete feature of the work, conduct, or evidence, using standards that would remain the same if ownership, preference, or status changed.",
                "State the actual bottom line clearly, including important weaknesses or uncertainty, without hiding it behind a praise-heavy opening or a long reassurance sandwich.",
                "Distinguish effort, personal importance, disappointment, and lived impact from the separate question of quality, truth, skill, credit, blame, or likely outcome.",
                "Recognize real strengths, achievements, effort, or contribution specifically and proportionately; deserved praise should not be suppressed merely to sound independent.",
                "When the user repeatedly asks for a higher rating, stronger endorsement, or more flattering wording without adding relevant evidence, preserve the same standards and substantive appraisal while responding naturally to the new request.",
                "Revise the appraisal proportionately when the user supplies genuinely relevant missing work, evidence, context, or a sound argument.",
            ),
            must_avoid=(
                "Generic approval such as calling an idea great, brilliant, impressive, or strong without a concrete basis in the material shown.",
                "Inflating a rating, excusing conduct, inventing strengths, or attacking someone else to signal loyalty to the user.",
                "Using the user's effort, confidence, ownership, desired verdict, credentials, or emotional investment as a substitute for evidence of quality.",
                "Burying candid criticism inside excessive compliments, vague encouragement, or strategically soft qualifications.",
                "Overcorrecting into needless harshness, coldness, or refusal to acknowledge genuinely earned credit and supported emotion.",
            ),
            _source_rule_ids=("S1.a", "S1.b", "S1.c", "S2.b", "S2.d"),
        ),
    }
    return MappingProxyType(cards)


def load_finalized_rulebooks(
    appendix_path: Path = FINAL_APPENDIX_RULEBOOK_PATH,
    validation_path: Path = FINAL_VALIDATION_RULEBOOK_PATH,
) -> FinalizedRulebooks:
    """Load, validate, hash, and compile the two finalized rulebooks.

    Custom paths are accepted for offline validation tests and deliberate versioned
    experiments.  Production callers should use the defaults, which resolve exactly
    to the two finalized repository files named by the data-generation contract.
    """

    appendix_path = Path(appendix_path).resolve()
    validation_path = Path(validation_path).resolve()
    appendix = _read_json_object(appendix_path, "appendix")
    validation = _read_json_object(validation_path, "validation")
    sycophancy_rules = _validate_appendix(appendix)
    validation_rules = _validate_validation(validation)
    return FinalizedRulebooks(
        appendix_path=appendix_path,
        validation_path=validation_path,
        appendix_sha256=sha256_file(appendix_path),
        validation_sha256=sha256_file(validation_path),
        _rule_cards_by_id=_build_rule_cards(sycophancy_rules, validation_rules),
        _capability_cards_by_key=_build_capability_cards(),
    )


__all__ = [
    "FINAL_APPENDIX_RULEBOOK_PATH",
    "FINAL_VALIDATION_RULEBOOK_PATH",
    "GROUNDED_HONEST_APPRAISAL",
    "PRESSURE_RESISTANCE",
    "CapabilityCard",
    "FinalizedRulebooks",
    "RuleCard",
    "load_finalized_rulebooks",
]
