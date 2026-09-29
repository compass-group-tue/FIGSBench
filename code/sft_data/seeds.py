"""Load and sample the balanced 24k HF-derived training-seed corpus."""

from __future__ import annotations

import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import DEFAULT_SEED_CORPUS, iter_jsonl

HF_SEED_TOTAL = 24_000
HF_SEED_DOMAIN_COUNT = 7
HF_SEED_MERGED_DOMAIN_ID = "science_technology"
HF_SEED_MERGED_DOMAIN_NAME = "Science & Technology"
HF_SEED_STANDARD_DOMAIN_COUNT = 3_000
HF_SEED_MERGED_DOMAIN_COUNT = 6_000


@dataclass(frozen=True)
class SeedRecord:
    training_seed_id: str
    record_id: str
    source: str
    source_stratum: str
    redacted_text: str
    word_count: int
    domain: dict[str, Any]
    setting: dict[str, Any]
    training_eligibility: str
    selection_score: float
    provenance: dict[str, Any]

    def prompt_view(self) -> dict[str, Any]:
        """Full internal provenance view persisted beside training data."""
        return {
            "training_seed_id": self.training_seed_id,
            "record_id": self.record_id,
            "source": self.source,
            "source_stratum": self.source_stratum,
            "redacted_text": self.redacted_text,
            "word_count": self.word_count,
            "domain": self.domain,
            "setting": self.setting,
            "training_eligibility": self.training_eligibility,
            "selection_score": self.selection_score,
            "provenance": self.provenance,
        }

    def model_view(self) -> dict[str, str]:
        """Minimal substantive seed content sent to a model."""
        return {"text": self.redacted_text}


class SeedPool:
    def __init__(self, path: Path = DEFAULT_SEED_CORPUS) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Training-seed corpus not found: {path}. The historical corpus "
                "is not redistributed; pass --seed-corpus with a JSONL file you "
                "are licensed to use (see "
                "code/source_corpus/data/example-seeds-v1/README.md for the schema)."
            )
        by_domain: dict[str, list[SeedRecord]] = defaultdict(list)
        by_id: dict[str, SeedRecord] = {}
        for raw in iter_jsonl(path):
            domain = raw.get("domain")
            setting = raw.get("setting")
            if not isinstance(domain, dict) or not domain.get("id") or not domain.get("name"):
                raise ValueError("Training seed has invalid domain")
            if not isinstance(setting, dict) or not setting.get("id") or not setting.get("name"):
                raise ValueError("Training seed has invalid setting")
            eligibility = str(raw.get("training_eligibility", ""))
            text = str(raw.get("redacted_text", "")).strip()
            seed_id = str(raw.get("training_seed_id", ""))
            if not text or not seed_id:
                raise ValueError("Training seed requires text and training_seed_id")
            record = SeedRecord(
                training_seed_id=seed_id,
                record_id=str(raw.get("record_id", "")),
                source=str(raw.get("source", "unknown")),
                source_stratum=str(raw.get("source_stratum", "")),
                redacted_text=text,
                word_count=int(raw.get("word_count", 0)),
                domain=dict(domain),
                setting=dict(setting),
                training_eligibility=eligibility,
                selection_score=float(raw.get("selection_score", 0.0)),
                provenance=dict(raw.get("provenance", {})),
            )
            if seed_id in by_id:
                raise ValueError(f"Duplicate training_seed_id: {seed_id}")
            by_id[seed_id] = record
            by_domain[str(domain["name"])].append(record)
        if not by_id:
            raise ValueError(f"Seed corpus is empty: {path}")
        # The canonical 500-slot release used the historical 24k corpus with an
        # exact per-domain balance. Custom corpora only need valid records and
        # coverage of the seed domains the plan uses; anything else changes
        # sampling density, so it warns instead of failing.
        canonical = (
            len(by_id) == HF_SEED_TOTAL
            and len(by_domain) == HF_SEED_DOMAIN_COUNT
            and all(
                len(records)
                == (
                    HF_SEED_MERGED_DOMAIN_COUNT
                    if name == HF_SEED_MERGED_DOMAIN_NAME
                    else HF_SEED_STANDARD_DOMAIN_COUNT
                )
                for name, records in by_domain.items()
            )
        )
        if not canonical:
            warnings.warn(
                f"SeedPool: non-canonical corpus at {path} "
                f"({len(by_id)} records, {len(by_domain)} domains); sampling "
                "density differs from the canonical 24k balance. New runs "
                "record their corpus sha256 in the run manifest.",
                UserWarning,
                stacklevel=2,
            )

        self.path = path.resolve()
        self.by_domain = dict(by_domain)
        self.by_id = by_id
        self._reuse_warned = False

    @property
    def domains(self) -> list[dict[str, str]]:
        return [
            {
                "id": records[0].domain["id"],
                "name": name,
            }
            for name, records in sorted(self.by_domain.items())
        ]

    def choose(self, domain_name: str, rng: random.Random) -> SeedRecord:
        records = self.by_domain.get(domain_name, [])
        if not records:
            raise ValueError(f"No training seeds for domain: {domain_name}")
        source_counts = Counter(record.source for record in records)
        weights = [1.0 / source_counts[record.source] for record in records]
        return rng.choices(records, weights=weights, k=1)[0]

    def choose_unique(
        self,
        domain_name: str,
        rng: random.Random,
        used: set[str],
        *,
        max_attempts: int = 1000,
    ) -> SeedRecord:
        """Return a seed not in ``used``, reusing one if the pool is exhausted.

        Draws are identical to repeated ``choose`` calls, so runs against the
        canonical corpus are unaffected; small custom corpora terminate with
        seed reuse (warned once) instead of looping forever.
        """
        seed = self.choose(domain_name, rng)
        attempts = 1
        while seed.training_seed_id in used and attempts < max_attempts:
            seed = self.choose(domain_name, rng)
            attempts += 1
        if seed.training_seed_id in used and not self._reuse_warned:
            self._reuse_warned = True
            warnings.warn(
                f"SeedPool: reusing seeds for domain {domain_name!r}; the "
                "corpus is smaller than the slot count. Scenarios remain "
                "newly written, but inspiration repeats.",
                UserWarning,
                stacklevel=2,
            )
        return seed

    def get(self, training_seed_id: str) -> SeedRecord:
        try:
            return self.by_id[training_seed_id]
        except KeyError as exc:
            raise ValueError(f"Unknown training_seed_id: {training_seed_id}") from exc


__all__ = [
    "HF_SEED_DOMAIN_COUNT",
    "HF_SEED_MERGED_DOMAIN_COUNT",
    "HF_SEED_MERGED_DOMAIN_ID",
    "HF_SEED_MERGED_DOMAIN_NAME",
    "HF_SEED_STANDARD_DOMAIN_COUNT",
    "HF_SEED_TOTAL",
    "SeedPool",
    "SeedRecord",
]
