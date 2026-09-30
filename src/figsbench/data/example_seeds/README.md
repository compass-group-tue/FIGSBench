# Example seed corpus

`seeds.jsonl` holds 28 short, original inspiration blurbs (4 per domain) written for this
release, so it can be shared freely. Use it to try the authoring pipeline:

```bash
python -m figsbench.generation --seed-corpus src/figsbench/data/example_seeds/seeds.jsonl --dry-run
python scripts/make_specs.py --seed 42 --output /tmp/my_specs.json
python -m figsbench.generation --seed-corpus src/figsbench/data/example_seeds/seeds.jsonl \
  --specs-file /tmp/my_specs.json --axis sycophancy --limit 2 --output-dir runs/example
```

The corpus is small, so slots reuse seeds and you will see a warning about it. The benchmark
itself was written from a 24,000-post corpus that is not redistributed.

## Format of your own corpus

One JSON object per line with:

- `training_seed_id`: a unique string (a 64-character hex id is recommended);
- `redacted_text`: the inspiration text. It is only loose inspiration: the person, facts and
  conversation are always newly written;
- `domain`: `{"id", "name"}`, covering all seven domains used here (`health_medicine`,
  `work_careers_organizations`, `finance_law`, `science_technology`, `intimate_relationships`,
  `family_friends_social_life`, `beliefs_culture_society`);
- `setting`: `{"id", "name"}`;
- optionally `source`, `source_stratum`, `word_count`, `training_eligibility`,
  `selection_score` and `provenance`. Record the real source and license of anything you use.
