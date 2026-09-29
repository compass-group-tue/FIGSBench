# Example seed corpus (synthetic, redistribution-safe)

`seeds.jsonl` holds 28 short, original inspiration blurbs (4 per seed domain)
written for this release. There is no third-party text in this file, so it can
be shared freely. Use it to try the authoring pipeline end to end:

```bash
cd code
python -m archetype_guided_benchmark.run_final --dry-run \
  --seed-corpus source_corpus/data/example-seeds-v1/seeds.jsonl
python scripts/make_specs.py --seed 42 --output /tmp/my_specs.json
python -m archetype_guided_benchmark.run_final --axis syc --limit 2 --workers 2 \
  --seed-corpus source_corpus/data/example-seeds-v1/seeds.jsonl \
  --specs-file /tmp/my_specs.json --run-id-syc example-syc
```

Because the pool is small, slots reuse seeds across runs; expect a
`UserWarning` noting the non-canonical corpus size. That warning is normal for
custom corpora. The canonical 500-slot release used the historical 24,000-row
corpus, which is not redistributed. Every run
records its own corpus sha256 in the run manifest (`training_seeds_sha256`).

## Custom corpus schema

A custom `--seed-corpus` JSONL needs one object per line with:

- `training_seed_id`: unique string (64-char hex recommended, so generated
  samples keep valid `source_seed_id` fingerprints);
- `redacted_text`: non-empty inspiration text (abstract inspiration only; the
  person, facts and conversation are always newly written);
- `domain`: `{"id", "name"}` — cover all seven seed domains used here
  (`health_medicine`, `work_careers_organizations`, `finance_law`,
  `science_technology`, `intimate_relationships`, `family_friends_social_life`,
  `beliefs_culture_society`), otherwise merged-domain slots cannot draw seeds;
- `setting`: `{"id", "name"}`;
- `source`, `source_stratum`, `word_count`, `training_eligibility`,
  `selection_score`, `provenance` (object; record the real source, license,
  and attribution status — a metadata tag alone does not prove
  redistribution rights).
