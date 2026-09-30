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
figsbench-validate .
python -m figsbench.generation --dry-run --seed-corpus src/figsbench/data/example_seeds/seeds.jsonl
```

Do not run paid authoring or evaluation calls as part of an ordinary test. New
tests should use deterministic fixtures or mocked clients.

## Pull requests

- Keep changes scoped and explain any effect on benchmark comparability.
- Add tests for behavior changes, especially resume behavior, model identity,
  judgment completeness, metric denominators, and schema validation.
- Do not change the 500 benchmark rows, the frozen judge prompts, or the
  metric without a new benchmark version and a note explaining the change.
- Run the validator, the test suite, and the dry-run planner before opening a
  pull request.

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
