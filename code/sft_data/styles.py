"""Prompt-generated free-flow texting-style descriptions.

Style quality is controlled by the generation prompt.  This module deliberately does
not infer style requirements with regexes or reject model text with stylistic heuristics.
Only the small structural contract needed by the pipeline is validated here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


STYLE_PROMPT_VERSION = "texting-style-catalog-v7"
STYLE_BATCH_SIZE = 16


def user_style_conformance_errors(
    style_description: str, user_messages: Sequence[str]
) -> list[str]:
    """Compatibility helper that checks structure, never textual style conformance.

    The transcript models receive the complete prose style description and are solely
    responsible for realizing it.  Returning content-derived errors here would recreate
    the brittle regex gate that this pipeline intentionally removed.
    """
    if not isinstance(style_description, str) or not style_description.strip():
        return ["texting_style_input: style description must be non-empty text"]
    if isinstance(user_messages, (str, bytes)):
        return ["texting_style_input: user messages must be a sequence of strings"]
    messages = list(user_messages)
    if not messages:
        return ["texting_style_input: at least one user message is required"]
    if any(not isinstance(message, str) or not message.strip() for message in messages):
        return ["texting_style_input: every user message must be non-empty text"]
    return []


def soften_style_description(value: str) -> str:
    """Compatibility normalizer; wording quality is handled in the prompt."""
    return " ".join(value.strip().split())


def validate_style(style: dict[str, Any]) -> None:
    """Validate only the style record's structural shape."""
    if set(style) != {"id", "name", "description"}:
        raise ValueError("Texting style must contain id, name, and description")
    for field in ("id", "name", "description"):
        if not isinstance(style[field], str) or not style[field].strip():
            raise ValueError(f"Texting style {field} must be non-empty text")


def validate_style_catalog(value: dict[str, Any], expected_count: int) -> None:
    """Validate count, record shape, and internal identifiers only."""
    styles = value.get("styles")
    if not isinstance(styles, list) or len(styles) != expected_count:
        raise ValueError(f"Style catalog must contain exactly {expected_count} styles")
    ids: set[str] = set()
    names: set[str] = set()
    for style in styles:
        if not isinstance(style, dict):
            raise ValueError("Each texting style must be an object")
        validate_style(style)
        if style["id"] in ids:
            raise ValueError("Texting style IDs must be unique")
        if style["name"] in names:
            raise ValueError("Texting style names must be unique")
        ids.add(style["id"])
        names.add(style["name"])


def style_batch_prompt(
    *, batch_index: int, batch_size: int, prior_styles: list[dict[str, str]]
) -> tuple[str, list[dict[str, str]]]:
    system = """Create detailed natural-language profiles for realistic human texting
in multi-turn assistant conversations. Each profile describes one coherent, flexible
voice rather than a bag of parameters. Explain typical message length and how it changes
across turns, cadence, capitalization, punctuation, paragraphing, fragments, hesitation,
self-correction, shorthand, typo frequency, emoji use, emotional directness, how the
person responds to the preceding answer, and which writing habits would feel out of
character.

Make the profiles visibly distinct in actual dialogue while keeping every voice readable
and plausible. Diversify along several dimensions at once. Include concise and expansive
writers; steady and bursty cadences; polished, ordinary, distracted, blunt, tentative,
technically fluent, and warmly conversational voices. Do not repeatedly fall back to the
same minimalist-lowercase-no-punctuation archetype. Within this batch, no two profiles
should share the same combination of length pattern, casing, punctuation, cadence, and
emotional presentation.

Describe tendencies, not a script. Give each writer natural variation across messages,
including how seriousness, haste, confusion, or relief can shift the surface form. Do not
assign a persona, demographic identity, worldview, topic preference, emotional agenda,
or content attitude. Do not caricature dialects or create roleplay characters. Each
description must be 80-220 words of flowing prose, not a list or parameter block.

Compare against every previously supplied description and create a genuinely different
texting pattern rather than renaming an earlier one. The prompt is the quality control:
carefully self-review diversity and realism before responding.

Return one JSON object with a `styles` array. Each item contains only `description`.
Return JSON only."""
    user = {
        "styles_requested": batch_size,
        "prior_style_descriptions": [style["description"] for style in prior_styles],
        "instruction": (
            "Create new, realistic profiles that are conceptually distinct from all "
            "prior descriptions."
        ),
    }
    return system, [
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)}
    ]


STYLE_PROMPT_VERSION_V8 = "texting-style-catalog-v8"


def style_batch_prompt_v8(
    *, batch_index: int, batch_size: int, prior_styles: list[dict[str, str]]
) -> tuple[str, list[dict[str, str]]]:
    system = """Create detailed natural-language profiles for realistic human texting
in multi-turn assistant conversations. Each profile describes one coherent, flexible
voice rather than a bag of parameters. Explain typical message length and how it changes
across turns, cadence, capitalization, punctuation, paragraphing, fragments, hesitation,
self-correction, shorthand, typo frequency, emoji use, emotional directness, how the
person responds to the preceding answer, and which writing habits would feel out of
character.

DEFAULT TO PLAIN DICTION: most voices must use short, ordinary, everyday words and
simple sentences. Nobody reaches for fancy, literary, or technical wording unless their
profile specifically requires it. A minority of voices, roughly one in five, may be
technically fluent professionals (engineer, clinician, analyst, lawyer, graduate
student), and even they use jargon only where the topic at hand demands it while
keeping everything else in plain words. Never write caricatures: no poetic or lyrical
voices, no logophiles hunting obscure words, no philosophers, no fairy-tale whimsy,
no regional-charm pastiches, no footnotes, no dot-point minimalists, no cyclical
narrators. Every voice must read as a real person typing on a phone.

Make the profiles visibly distinct in actual dialogue while keeping every voice readable
and plausible. Diversify along several dimensions at once. Include concise and expansive
writers; steady and bursty cadences; polished, ordinary, distracted, blunt, tentative,
warm, and dryly funny voices. Do not repeatedly fall back to the same
minimalist-lowercase-no-punctuation archetype. Within this batch, no two profiles
should share the same combination of length pattern, casing, punctuation, cadence, and
emotional presentation.

Describe tendencies, not a script. Give each writer natural variation across messages,
including how seriousness, haste, confusion, or relief can shift the surface form. Do not
assign a persona, demographic identity, worldview, topic preference, emotional agenda,
or content attitude. Do not caricature dialects or create roleplay characters. Each
description must be 80-220 words of flowing prose, not a list or parameter block.

Compare against every previously supplied description and create a genuinely different
texting pattern rather than renaming an earlier one. The prompt is the quality control:
carefully self-review diversity and realism before responding.

Return one JSON object with a `styles` array. Each item contains only `description`.
Return JSON only."""
    user = {
        "styles_requested": batch_size,
        "prior_style_descriptions": [style["description"] for style in prior_styles],
        "instruction": (
            "Create new, realistic profiles that are conceptually distinct from all "
            "prior descriptions."
        ),
    }
    return system, [
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)}
    ]


def generate_style_catalog_v8(
    client: Any,
    count: int = 64,
    *,
    temperature: float = 0.7,
    max_tokens: int = 16000,
    batch_size: int = 8,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """V8 variant of generate_style_catalog using the plain-diction prompt."""
    import sft_data.styles as _self

    original = _self.style_batch_prompt
    _self.style_batch_prompt = style_batch_prompt_v8
    try:
        result = _self.generate_style_catalog(
            client,
            count=count,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )
    finally:
        _self.style_batch_prompt = original
    result["prompt_version"] = STYLE_PROMPT_VERSION_V8
    return result


def generate_style_catalog(
    client: Any,
    count: int = 64,
    *,
    temperature: float = 1.0,
    max_tokens: int = 6500,
    batch_size: int = STYLE_BATCH_SIZE,
    max_batch_attempts: int = 6,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    if count < batch_size or count % batch_size:
        raise ValueError(f"style count must be a multiple of {batch_size}")
    styles: list[dict[str, str]] = []
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        styles = json.loads(Path(checkpoint_path).read_text())["styles"]
    for batch_index in range(len(styles) // batch_size, count // batch_size):
        last_error: Exception | None = None
        for _ in range(max_batch_attempts):
            system, messages = style_batch_prompt(
                batch_index=batch_index + 1,
                batch_size=batch_size,
                prior_styles=styles,
            )
            try:
                value = client.complete_json(
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    attempts=3,
                )
                batch = value.get("styles")
                if not isinstance(batch, list) or len(batch) != batch_size:
                    raise ValueError("Model returned the wrong number of texting styles")
                proposed: list[dict[str, str]] = []
                for raw in batch:
                    if (
                        not isinstance(raw, dict)
                        or not isinstance(raw.get("description"), str)
                        or not raw["description"].strip()
                    ):
                        raise ValueError(
                            "Each generated style must contain a non-empty description"
                        )
                    sequence = len(styles) + len(proposed) + 1
                    proposed.append(
                        {
                            "id": f"style-{sequence:03d}",
                            "name": f"Texting profile {sequence:03d}",
                            "description": soften_style_description(raw["description"]),
                        }
                    )
                styles.extend(proposed)
                if checkpoint_path is not None:
                    Path(checkpoint_path).write_text(
                        json.dumps({"styles": styles}, indent=1)
                    )
                break
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Could not generate structurally valid texting-style batch "
                f"{batch_index + 1}: {last_error}"
            ) from last_error
    result = {
        "schema_version": 1,
        "prompt_version": STYLE_PROMPT_VERSION,
        "styles": styles,
    }
    validate_style_catalog(result, count)
    return result


__all__ = [
    "STYLE_BATCH_SIZE",
    "STYLE_PROMPT_VERSION",
    "generate_style_catalog",
    "soften_style_description",
    "style_batch_prompt",
    "user_style_conformance_errors",
    "validate_style",
    "validate_style_catalog",
]
