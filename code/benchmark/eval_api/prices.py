"""Per-model API prices ($ per 1M tokens: input / cached-input / output).

Pass service_tier="flex" for OpenAI flex routes (50% off all rates).

Reasoning/thinking tokens bill at the output rate. Local (`local-*`) models
are self-hosted: cost 0. Unknown models: tokens logged, cost None.
"""
from __future__ import annotations

from typing import Any

# (match-substring, input, cached, output)
PRICE_TABLE: list[tuple[str, float, float, float]] = [
    ("deepseek-v4.1-flash", 0.30, 0.006, 1.20),
    ("deepseek-v4-flash", 0.30, 0.006, 1.20),
    ("gemini-3.8-flash", 0.75, 0.075, 3.75),
    ("muse-spark", 1.25, 0.15, 4.25),
    ("glm-5.3", 1.40, 0.26, 4.40),
    ("glm-5", 1.40, 0.26, 4.40),
    ("claude-sonnet-5", 2.00, 0.20, 10.00, 2.50),
    ("claude-haiku-4.5", 1.00, 0.10, 5.00, 1.25),
    ("grok-4", 2.00, 0.50, 6.00),
    ("kimi-k3", 3.00, 0.30, 15.00),
    ("gpt-5.6-sol", 4.00, 0.40, 20.00),
    ("gpt-5.6-luna", 1.50, 0.15, 12.00),
    ("gpt-6-astra", 10.00, 1.00, 50.00),
    ("claude-fable-5.1", 10.00, 1.00, 50.00, 12.50),
]


def lookup(model: str) -> tuple[float, float, float, float] | None:
    name = (model or "").strip().lower()
    if name.startswith("local-"):
        return (0.0, 0.0, 0.0, 0.0)
    for row in PRICE_TABLE:
        sub, i, c, o = row[0], row[1], row[2], row[3]
        w = row[4] if len(row) > 4 else i
        if sub in name:
            return (i, c, o, w)
    return None


def cost_usd(
    model: str, usage: dict[str, Any], service_tier: str | None = None
) -> float | None:
    rates = lookup(model)
    if rates is None:
        return None
    inp, cached, outp, write = rates
    if (service_tier or "").strip().lower() == "flex":
        inp, cached, outp, write = inp / 2, cached / 2, outp / 2, write / 2
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    cached_tok = int(usage.get("cached_tokens", 0) or 0)
    written = int(usage.get("cache_write_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    reasoning = int(usage.get("reasoning_tokens", 0) or 0)
    fresh = max(prompt - cached_tok - written, 0)
    return (fresh * inp + written * write + cached_tok * cached + (completion + reasoning) * outp) / 1e6
