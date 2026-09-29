"""Deterministic planning for rulebook-grounded anti-sycophancy SFT data."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Sequence


DEFAULT_TARGET = 150
DEFAULT_SEED = 20260811
DEFAULT_STYLE_COUNT = 64

# The 100/50 margin is a primary generation emphasis, not a pair of exclusive
# behavioral lanes.  Validation remains compiled into every behavior card as a
# cross-cutting calibration constraint.  These focuses are generation tasks, not
# benchmark labels, and models receive only their prose descriptions.
FOCUS_WEIGHTS = {
    "sustained_pressure": 1 / 3,
    "grounded_appraisal": 1 / 3,
    "pressure_then_update": 1 / 6,
    "ordinary_calibrated": 1 / 6,
}

FOCUS_DESCRIPTIONS = {
    "sustained_pressure": (
        "The first assistant answer states a clear, warranted position. At least two "
        "later user turns increase the interpersonal or practical pressure while adding "
        "no decision-relevant reason. Every later assistant answer remains substantively "
        "stable without becoming repetitive, cold, or defensive."
    ),
    "grounded_appraisal": (
        "The user supplies or fully describes their own idea, work, plan, argument, or "
        "conduct and seeks an evaluation. The assistant judges it by visible criteria, "
        "gives only earned and specific credit, states material weaknesses plainly, and "
        "does not inflate the appraisal when the user presses for stronger approval. At "
        "least two later turns should test that independence naturally."
    ),
    "pressure_then_update": (
        "The assistant first gives a clear, warranted answer, then holds it through at "
        "least one pressure-only turn. A later turn supplies genuinely relevant new "
        "evidence, reasoning, or a changed constraint. The assistant then updates only as "
        "much as that new reason warrants and identifies the concrete change."
    ),
    "ordinary_calibrated": (
        "Use an ordinary low-pressure exchange that demonstrates independent, calibrated "
        "judgment without manufacturing conflict. The assistant may agree, praise, "
        "recognize, or recommend confidently when the visible record warrants it, while "
        "remaining specific and honest rather than adding automatic approval."
    ),
}

LOCUS_DESCRIPTIONS = {
    "world_claim": (
        "The central judgment concerns what is true, likely, caused, intended, or "
        "supported by the available evidence."
    ),
    "appraisal": (
        "The central judgment concerns quality, skill, credit, blame, fairness, or the "
        "merit of a user-proposed artifact, idea, plan, or action."
    ),
    "advice": (
        "The central judgment concerns what the user should do and how strong or "
        "consequential the recommended response should be."
    ),
}

# Within the two non-appraisal-specific focuses, vary the substantive locus.  The
# grounded-appraisal focus is always an appraisal task by construction.
LOCUS_WEIGHTS_BY_FOCUS = {
    "sustained_pressure": {
        "world_claim": 0.55,
        "appraisal": 0.10,
        "advice": 0.35,
    },
    "grounded_appraisal": {"appraisal": 1.0},
    "pressure_then_update": {
        "world_claim": 0.48,
        "appraisal": 0.12,
        "advice": 0.40,
    },
    "ordinary_calibrated": {
        "world_claim": 0.20,
        "appraisal": 0.60,
        "advice": 0.20,
    },
}

PRESSURE_TURN_WEIGHTS = {3: 0.20, 4: 0.40, 5: 0.40}
ORDINARY_TURN_WEIGHTS = {2: 2 / 3, 3: 1 / 3}
# Public combined vocabulary retained for callers that only need valid turn values.
TURN_WEIGHTS = {2: 0.10, 3: 0.20, 4: 0.35, 5: 0.35}

APPRAISAL_CALIBRATION_DESCRIPTIONS = {
    "mostly_weak": (
        "The visible work is mostly weak or not yet ready. Any positive credit must be "
        "narrow and concrete; the overall appraisal should remain candid."
    ),
    "mixed": (
        "The visible work has real strengths and real limitations. Do not force a praise "
        "sandwich; make the overall appraisal match which considerations matter most."
    ),
    "mostly_strong": (
        "The visible work is genuinely strong by the supplied criteria and deserves "
        "specific positive evaluation, while any remaining limitation is stated at its "
        "actual importance. Independence must not become reflexive negativity."
    ),
}


# These are free-flow descriptions of how a real exchange develops.  They are not
# pressure-source labels and are never used as benchmark categories.
# Locus-specific pools prevent a conversational framing from silently changing the
# substantive question.  The flattened aliases remain public for reporting and tests.
_TARGETED_SHAPE_BASES_BY_LOCUS: dict[str, tuple[str, ...]] = {
    "world_claim": (
        (
            "Start with a self-contained record that supports a bounded factual "
            "interpretation. Later turns emphasize the evidence favoring the user's preferred "
            "conclusion while the assistant continues to weigh the complete record."
        ),
        (
            "Center the exchange on whether a named source, expert, or witness has relevant "
            "access, independence, and reasoning. The user leans on confidence or credentials; "
            "the assistant calibrates what that source can actually establish."
        ),
        (
            "Use an observed sequence or association and ask what caused it. Follow-ups press "
            "for a definite causal story, while any genuine update remains limited to newly "
            "visible evidence about mechanism or dependence."
        ),
        (
            "Ask for a prediction from supplied measurements or observations. Later turns seek "
            "greater certainty or a sharper forecast, while the assistant keeps confidence and "
            "live alternatives calibrated to the evidence then visible."
        ),
        (
            "Give a compact factual claim or interpretation and ask whether the visible record "
            "supports it. Later turns request a stronger formulation, but the central task "
            "remains the truth and confidence warranted by that record."
        ),
        (
            "Begin after a concrete outcome and ask what it proves about an earlier explanation "
            "or prediction. The user treats success or failure as decisive; the assistant "
            "separates the observed result from unsupported causal or general conclusions."
        ),
    ),
    "appraisal": (
        (
            "Present a complete visible artifact and ask for an overall quality judgment using "
            "criteria available from the start. Later turns dispute one feature or ask whether "
            "it changes the same appraisal."
        ),
        (
            "Use complete visible work whose owner wants a favorable rating. Follow-ups "
            "emphasize effort, intent, or emotional investment, while the assistant keeps those "
            "separate from the artifact's quality under the assigned calibration."
        ),
        (
            "Use visible work whose quality matches the assigned calibration and let later "
            "turns seek broader praise or a person-level virtue claim. Preserve only the "
            "feature-specific credit the artifact earns."
        ),
        (
            "Use a complete artifact with multiple inspectable features. Later turns ask to "
            "discount an inconvenient weakness or overemphasize a strength, while the overall "
            "assessment continues to follow the assigned calibration."
        ),
        (
            "Supply a bounded conduct or contribution record and ask about credit, fairness, or "
            "blame. Follow-ups seek a more favorable moral or competence judgment while the "
            "assistant applies the same visible standards."
        ),
        (
            "Start with a complete artifact and a requested rating or comparison. Later turns "
            "zoom into a disputed sentence, number, assumption, or revision while every score "
            "continues to use the same visible quality criteria."
        ),
    ),
    "advice": (
        (
            "Present two feasible actions with the user's goal, values, constraints, and known "
            "tradeoffs visible from the start. Later turns favor one option or press for a more "
            "categorical recommendation."
        ),
        (
            "Let the user have already committed time, money, or reputation to one course. The "
            "awkwardness of changing direction grows across turns, but advice stays tied to "
            "future costs, feasibility, and the user's stated objective."
        ),
        (
            "Use an urgent but self-contained action choice. Follow-ups request a faster, harsher, "
            "or less reversible response; the assistant calibrates intensity to the visible risk "
            "and offers a useful proportionate step."
        ),
        (
            "Begin with a supported course of action and then turn it into a message, sequence, "
            "or implementation step. Later requests push the wording beyond what is useful for "
            "the user's stated goal, while the advice remains practical."
        ),
        (
            "Frame a concrete resource, scope, or deadline tradeoff. The user wants every goal "
            "preserved, but the assistant recommends among visible options and makes the real "
            "implementation tradeoff explicit."
        ),
        (
            "Use a reasonable personal boundary or preference with visible interpersonal "
            "consequences. Later turns ask the assistant to take over the decision or guarantee "
            "another person's response; advice remains useful while leaving control with the user."
        ),
    ),
}

_TARGETED_TRAJECTORY_NOTE = (
    "The assigned reason trajectory alone determines whether a later fact, artifact revision, "
    "or changed constraint updates the judgment."
)

CONVERSATION_SHAPES_BY_LOCUS: dict[str, tuple[str, ...]] = {
    locus: tuple(f"{shape} {_TARGETED_TRAJECTORY_NOTE}" for shape in shapes)
    for locus, shapes in _TARGETED_SHAPE_BASES_BY_LOCUS.items()
}

CONVERSATION_SHAPES = tuple(
    shape
    for locus in LOCUS_DESCRIPTIONS
    for shape in CONVERSATION_SHAPES_BY_LOCUS[locus]
)

ORDINARY_SHAPES_BY_LOCUS = {
    "world_claim": (
        "Ask a straightforward evidence question whose favorable answer follows from the visible record. Later turns request one implication and one confidence boundary without manufacturing disagreement.",
        "Use a source-reliability question with visible access and independence information. The user accepts the answer and asks how much the source establishes.",
        "Begin after a reported outcome and ask what it does and does not show about a factual explanation, keeping the exchange collaborative.",
        "Start with a well-supported conclusion and a request for a concise factual explanation, followed by a normal example or application.",
    ),
    "appraisal": (
        "Supply a complete genuinely strong artifact and ask an open-ended quality question. Later turns ask how to preserve its best visible feature without fishing for a higher verdict.",
        "Center the exchange on an effective visible draft. Later turns request one polish pass and a shorter version while preserving earned criterion-specific praise.",
        "Provide a concrete contribution record that genuinely deserves credit, then ask for a fair appraisal of that contribution without global character praise.",
        "Present two complete artifacts and visible quality criteria that support a clear comparison. Follow-ups probe one feature without changing the criteria.",
    ),
    "advice": (
        "Use an ordinary planning request with visible goals and constraints. The assistant recommends one supported option and later helps with a small implementation step.",
        "Present two feasible options and enough personal criteria for one to be a reasonable preference. Follow-ups ask about a tradeoff and implementation.",
        "Use a reasonable boundary or preference and ask how to communicate it. Later turns refine tone and timing while the user retains the decision.",
        "Begin with a supported practical direction and let later turns turn it into ordering, wording, or a reversible next step without manufactured resistance.",
    ),
}

ORDINARY_CONVERSATION_SHAPES = tuple(
    shape
    for locus in LOCUS_DESCRIPTIONS
    for shape in ORDINARY_SHAPES_BY_LOCUS[locus]
)


def _largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    if total < 0 or not weights or sum(weights) <= 0:
        raise ValueError("Invalid allocation")
    normalized = [weight / sum(weights) for weight in weights]
    raw = [total * weight for weight in normalized]
    counts = [math.floor(value) for value in raw]
    order = sorted(
        range(len(raw)), key=lambda index: (-(raw[index] - counts[index]), index)
    )
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _schedule(
    values: Sequence[Any],
    total: int,
    *,
    seed: int,
    weights: Sequence[float] | None = None,
) -> list[Any]:
    if not values:
        raise ValueError("Cannot schedule an empty value set")
    counts = _largest_remainder(total, weights or [1.0] * len(values))
    result = [
        deepcopy(value) for value, count in zip(values, counts) for _ in range(count)
    ]
    random.Random(seed).shuffle(result)
    return result


def _locus_schedule(focuses: list[str], *, seed: int) -> list[str]:
    result = [""] * len(focuses)
    for offset, focus in enumerate(FOCUS_WEIGHTS):
        indices = [index for index, value in enumerate(focuses) if value == focus]
        weights = LOCUS_WEIGHTS_BY_FOCUS[focus]
        loci = _schedule(
            list(weights),
            len(indices),
            seed=seed + offset,
            weights=list(weights.values()),
        )
        for index, locus in zip(indices, loci):
            result[index] = locus
    if any(not value for value in result):  # pragma: no cover - defensive
        raise RuntimeError("Failed to assign a decision locus")
    return result


def _turn_schedule(focuses: list[str], *, seed: int) -> list[int]:
    result = [0] * len(focuses)
    ordinary = [
        index for index, focus in enumerate(focuses) if focus == "ordinary_calibrated"
    ]
    pressure = [index for index in range(len(focuses)) if index not in set(ordinary)]
    ordinary_turns = _schedule(
        list(ORDINARY_TURN_WEIGHTS),
        len(ordinary),
        seed=seed,
        weights=list(ORDINARY_TURN_WEIGHTS.values()),
    )
    pressure_turns = _schedule(
        list(PRESSURE_TURN_WEIGHTS),
        len(pressure),
        seed=seed + 1,
        weights=list(PRESSURE_TURN_WEIGHTS.values()),
    )
    for index, turns in zip(ordinary, ordinary_turns):
        result[index] = turns
    for index, turns in zip(pressure, pressure_turns):
        result[index] = turns
    return result


def _appraisal_calibration_schedule(
    focuses: list[str], loci: list[str], *, seed: int
) -> dict[int, str]:
    appraisal_indices = [
        index for index, locus in enumerate(loci) if locus == "appraisal"
    ]
    ordinary = [
        index
        for index in appraisal_indices
        if focuses[index] == "ordinary_calibrated"
    ]
    remaining = [index for index in appraisal_indices if index not in set(ordinary)]
    count = len(appraisal_indices)
    base, remainder = divmod(count, 3)
    target_counts = {
        "mostly_weak": base,
        "mixed": base + (1 if remainder > 1 else 0),
        "mostly_strong": base + (1 if remainder > 0 else 0),
    }
    if len(ordinary) > target_counts["mostly_strong"]:
        raise ValueError("ordinary appraisal controls exceed the strong-appraisal margin")
    remaining_values = [
        value
        for value in APPRAISAL_CALIBRATION_DESCRIPTIONS
        for _ in range(
            target_counts[value]
            - (len(ordinary) if value == "mostly_strong" else 0)
        )
    ]
    random.Random(seed).shuffle(remaining_values)
    result = {index: "mostly_strong" for index in ordinary}
    result.update(dict(zip(remaining, remaining_values)))
    if len(result) != len(appraisal_indices):  # pragma: no cover - defensive
        raise RuntimeError("Failed to assign appraisal calibration")
    return result


def _conversation_shape_schedule(
    focuses: list[str], loci: list[str], *, seed: int
) -> list[str]:
    result = [""] * len(focuses)
    ordinary = [index for index, focus in enumerate(focuses) if focus == "ordinary_calibrated"]
    ordinary_set = set(ordinary)
    targeted = [index for index in range(len(focuses)) if index not in ordinary_set]
    for offset, locus in enumerate(LOCUS_DESCRIPTIONS):
        ordinary_indices = [index for index in ordinary if loci[index] == locus]
        ordinary_shapes = _schedule(
            ORDINARY_SHAPES_BY_LOCUS[locus],
            len(ordinary_indices),
            seed=seed + offset,
        )
        for index, shape in zip(ordinary_indices, ordinary_shapes):
            result[index] = shape
        targeted_indices = [index for index in targeted if loci[index] == locus]
        targeted_shapes = _schedule(
            CONVERSATION_SHAPES_BY_LOCUS[locus],
            len(targeted_indices),
            seed=seed + 10 + offset,
        )
        for index, shape in zip(targeted_indices, targeted_shapes):
            result[index] = shape
    if any(not shape for shape in result):  # pragma: no cover - defensive
        raise RuntimeError("Failed to assign a conversation shape")
    return result


def build_assignments(
    domains: list[dict[str, str]],
    target: int = DEFAULT_TARGET,
    *,
    seed: int = DEFAULT_SEED,
    style_count: int = DEFAULT_STYLE_COUNT,
) -> list[dict[str, Any]]:
    """Build the immutable assignment plan for one versioned generation run."""
    if target < 10 or target % 10:
        raise ValueError("target must be a multiple of ten and at least ten")
    if style_count < 16 or style_count % 16:
        raise ValueError("style_count must be a multiple of 16 and at least 16")
    if not domains:
        raise ValueError("At least one domain is required")

    focus_names = list(FOCUS_WEIGHTS)
    focuses = _schedule(
        focus_names,
        target,
        seed=seed + 1,
        weights=list(FOCUS_WEIGHTS.values()),
    )
    loci = _locus_schedule(focuses, seed=seed + 10)
    domain_schedule = _schedule(domains, target, seed=seed + 20)
    turns = _turn_schedule(focuses, seed=seed + 21)
    shapes = _conversation_shape_schedule(focuses, loci, seed=seed + 22)

    calibration_by_index = _appraisal_calibration_schedule(
        focuses, loci, seed=seed + 23
    )

    # Shuffle an even schedule instead of drawing with replacement.  At target=150,
    # every one of the 64 generated descriptions is used two or three times.
    style_schedule = _schedule(
        list(range(style_count)), target, seed=seed + 24
    )
    assignments: list[dict[str, Any]] = []
    for index in range(target):
        assignments.append(
            {
                "schema_version": 3,
                "id": f"SFT-{index + 1:04d}",
                "behavior_focus": focuses[index],
                "decision_locus": loci[index],
                "appraisal_calibration": calibration_by_index.get(index),
                "domain": domain_schedule[index],
                "user_turn_count": turns[index],
                "texting_style_index": style_schedule[index],
                "conversation_shape": shapes[index],
            }
        )
    return assignments


def assignment_model_view(assignment: dict[str, Any]) -> dict[str, Any]:
    """Return only substantive prose requirements suitable for a model prompt."""
    value: dict[str, Any] = {
        "domain": assignment["domain"]["name"],
        "user_turn_count": assignment["user_turn_count"],
        "target_behavior": FOCUS_DESCRIPTIONS[assignment["behavior_focus"]],
        "judgment_kind": LOCUS_DESCRIPTIONS[assignment["decision_locus"]],
        "conversation_shape": assignment["conversation_shape"],
    }
    calibration = assignment.get("appraisal_calibration")
    if calibration is not None:
        value["appraisal_calibration"] = APPRAISAL_CALIBRATION_DESCRIPTIONS[
            calibration
        ]
    return value


def required_reason_trajectory(assignment: dict[str, Any]) -> list[str]:
    """Return the deterministic user-turn reason sequence for an assignment."""

    turns = int(assignment["user_turn_count"])
    focus = assignment["behavior_focus"]
    if focus == "ordinary_calibrated":
        return ["initial_basis", *(["same_basis_followup"] * (turns - 1))]
    if turns < 3:
        raise ValueError(f"{focus} requires at least three user turns")
    if focus == "pressure_then_update":
        return [
            "initial_basis",
            *(["pressure_only"] * (turns - 2)),
            "new_relevant_reason",
        ]
    if focus in {"sustained_pressure", "grounded_appraisal"}:
        return ["initial_basis", *(["pressure_only"] * (turns - 1))]
    raise ValueError(f"unknown behavior focus: {focus}")


def _load_rule_chunks(*, appendix_path=None, validation_path=None) -> list[dict[str, Any]]:
    """Load atomic rule chunks from frozen rulebooks (S1.a-S2.d + V1-V3)."""
    import json as _json
    from pathlib import Path as _Path

    from .rulebooks import FINAL_APPENDIX_RULEBOOK_PATH, FINAL_VALIDATION_RULEBOOK_PATH

    appendix = _Path(appendix_path) if appendix_path else _Path(FINAL_APPENDIX_RULEBOOK_PATH)
    validation = _Path(validation_path) if validation_path else _Path(FINAL_VALIDATION_RULEBOOK_PATH)
    data_a = _json.loads(appendix.read_text(encoding="utf-8"))
    data_v = _json.loads(validation.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    for rule in data_a.get("sycophancy", []) + data_a.get("calibrated_validation", []):
        chunks.append(
            {
                "rule_id": rule["rule_id"],
                "title": rule["title"],
                "tier": rule["tier"],
                "description": rule["description"],
                "example": rule["example"],
                "neighboring_boundary": rule["neighboring_boundary"],
            }
        )
    for rule in data_v.get("calibrated_validation", []):
        chunks.append(
            {
                "rule_id": rule["rule_id"],
                "title": rule["title"],
                "tier": rule["tier"],
                "description": rule["description"],
                "example": rule["example"],
                "neighboring_boundary": rule["neighboring_boundary"],
            }
        )
    # dedupe by rule_id, keep order
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in chunks:
        if c["rule_id"] not in seen:
            seen.add(c["rule_id"])
            uniq.append(c)
    if not uniq:
        raise ValueError("No rule chunks loaded")
    return uniq


def build_assignments_v14(
    domains: list[dict[str, str]],
    target: int = DEFAULT_TARGET,
    *,
    seed: int = DEFAULT_SEED,
    style_count: int = DEFAULT_STYLE_COUNT,
    appendix_path=None,
    validation_path=None,
) -> list[dict[str, Any]]:
    """v14: rule-chunk-driven assignments — no explicit locus/focus.

    Each slot samples one atomic rule chunk (S1.a-S2.d, V1-V3) uniformly;
    locus/focus are inferred dynamically from the rule chunk.
    """
    if target < 10 or target % 10:
        raise ValueError("target must be a multiple of ten and at least ten")
    if style_count < 16 or style_count % 16:
        raise ValueError("style_count must be a multiple of 16 and at least 16")
    if not domains:
        raise ValueError("At least one domain is required")

    chunks = _load_rule_chunks(appendix_path=appendix_path, validation_path=validation_path)
    # uniform rule sampling, shuffled deterministically
    rule_indices = _schedule(list(range(len(chunks))), target, seed=seed + 1)
    domain_schedule = _schedule(domains, target, seed=seed + 20)
    # keep same turn distribution as before but independent of focus
    turns = _schedule(list(TURN_WEIGHTS), target, seed=seed + 21, weights=list(TURN_WEIGHTS.values()))
    # uniform shape sampling
    all_shapes = list(CONVERSATION_SHAPES) + list(ORDINARY_CONVERSATION_SHAPES)
    shapes = _schedule(all_shapes, target, seed=seed + 22)
    style_schedule = _schedule(list(range(style_count)), target, seed=seed + 24)

    assignments: list[dict[str, Any]] = []
    for index in range(target):
        rc = chunks[rule_indices[index]]
        assignments.append(
            {
                "schema_version": 4,
                "id": f"SFT-{index + 1:04d}",
                "benchmark_axis": "sycophancy",
                "rule_chunk": rc,
                "domain": domain_schedule[index],
                "user_turn_count": turns[index],
                "texting_style_index": style_schedule[index],
                "conversation_shape": shapes[index],
            }
        )
    return assignments


def assignment_model_view_v14(assignment: dict[str, Any]) -> dict[str, Any]:
    """Prose view for v14 — only rule chunk, domain, turns, shape."""
    rc = assignment.get("rule_chunk", {})
    return {
        "domain": assignment["domain"]["name"],
        "user_turn_count": assignment["user_turn_count"],
        "rule_chunk": {
            "rule_id": rc.get("rule_id"),
            "title": rc.get("title"),
            "description": rc.get("description"),
            "example": rc.get("example"),
            "neighboring_boundary": rc.get("neighboring_boundary"),
        },
        "conversation_shape": assignment["conversation_shape"],
    }


__all__ = [
    "APPRAISAL_CALIBRATION_DESCRIPTIONS",
    "CONVERSATION_SHAPES",
    "CONVERSATION_SHAPES_BY_LOCUS",
    "ORDINARY_CONVERSATION_SHAPES",
    "ORDINARY_SHAPES_BY_LOCUS",
    "DEFAULT_SEED",
    "DEFAULT_STYLE_COUNT",
    "DEFAULT_TARGET",
    "FOCUS_DESCRIPTIONS",
    "FOCUS_WEIGHTS",
    "LOCUS_DESCRIPTIONS",
    "TURN_WEIGHTS",
    "assignment_model_view",
    "assignment_model_view_v14",
    "build_assignments",
    "build_assignments_v14",
    "required_reason_trajectory",
]
