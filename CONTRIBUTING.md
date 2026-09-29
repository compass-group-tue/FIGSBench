# Contributing

Contributions that improve evaluator correctness, reproducibility,
documentation, portability, and test coverage are welcome.

## Development setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
sycbench-validate .
python -m archetype_guided_benchmark.run_final --dry-run --seed-corpus code/source_corpus/data/example-seeds-v1/seeds.jsonl
```

Do not run paid authoring or evaluation calls as part of an ordinary test. New
tests should use deterministic fixtures or mocked clients.

## Pull requests

- Keep changes scoped and explain any effect on benchmark comparability.
- Add tests for behavior changes, especially resume behavior, model identity,
  judgment completeness, metric denominators, and schema validation.
- Do not modify the canonical 500 rows, frozen judge prompts, judge locks, or
  official metric without an explicit versioned benchmark change and migration
  note.
- Run the release validator, test suite, dry-run planner, and checksum checks.
- Regenerate `MANIFEST.sha256` only as a deliberate final release step and
  review every changed entry.

## Data contributions

Do not contribute scraped or third-party text without documented authority to
use and redistribute it. Record the source URL, author/creator where required,
exact license and version, access date, modifications, and attribution text.
Never assume that public accessibility or an upstream metadata tag grants
redistribution rights. Remove secrets and direct personal identifiers, but do
not describe redaction alone as consent or anonymization.

By intentionally submitting a code contribution, you agree that it may be
distributed under Apache-2.0 as described in `LICENSE`. Project-authored data
and prompt contributions must be compatible with `DATA_LICENSE.md`.
