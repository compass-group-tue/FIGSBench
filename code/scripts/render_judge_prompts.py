"""Render the exact judge requests (system + user message) for the four locked hr-v11 stages.

Added for the public release (not part of the original pipeline). It configures
the judge runner exactly as `rule_guided_judging.hr_v11_judge.run_locked_judge`
does for each stage and renders the request for one sample's reference
transcript. No API calls are made.

Stages (every transcript is judged on both axes):
  1. syc-score  (score-only, syc-score-v1) -> sycophancy score 1-4
  2. cv-score   (score-only, cv-score-v1)  -> under-validation score 1-4
  3. syc-rules  (syc-rules-v1, conditioned on the locked syc score) -> broken S-rules
  4. cv-rules   (1.0,  conditioned on the locked CV score)  -> broken V-rules

The rules stages receive the stage-1/2 score as LOCKED_SCORE; this renderer
fills it with a placeholder value (--locked-score) because no judge is called.

    cd code
    python scripts/render_judge_prompts.py FINAL-SYC-001 --out ../prompts/judge_rendered
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
RELEASE = CODE.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample_id")
    ap.add_argument("--locked-score", type=int, default=2)
    ap.add_argument("--out", type=Path, default=RELEASE / "prompts/judge_rendered")
    args = ap.parse_args()

    sample = next((json.loads(l) for l in open(RELEASE / "data/benchmark_500.jsonl")
                   if json.loads(l)["id"] == args.sample_id), None)
    if sample is None:
        raise SystemExit(f"unknown sample id {args.sample_id}")
    transcript = [{"turn": t["turn"], "role": t["role"], "content": t["content"]}
                  for t in sample["transcript"]["turns"]]

    from rule_guided_judging import run as base
    from rule_guided_judging import hr_v11_cv_rules, hr_v11_cv_score, hr_v11_syc_rules, hr_v11_syc_score

    locked = {"score": args.locked_score, "quote": "", "description": ""}
    rendered = {}
    for name, module, axis in (("1_syc_score", hr_v11_syc_score, "sycophancy"),
                               ("2_cv_score", hr_v11_cv_score, "calibrated_validation")):
        module.configure_base_runner()
        contract = base.output_contract({})
        rendered[name] = (base.load_system_prompt(),
                          base.build_user_prompt(rules_by_axis={}, contract=contract, transcript=transcript))
    for name, module in (("3_syc_rules", hr_v11_syc_rules), ("4_cv_rules", hr_v11_cv_rules)):
        module.configure_rules_runner()
        rendered[name] = (base.load_system_prompt(),
                          module.build_rules_user_prompt(transcript=transcript, locked=locked))

    out = args.out / args.sample_id
    out.mkdir(parents=True, exist_ok=True)
    for name, (system, user) in rendered.items():
        (out / f"{name}.system.txt").write_text(system)
        (out / f"{name}.user.txt").write_text(user)
    print(f"wrote {len(rendered)} judge stages for {args.sample_id} -> {out}")


if __name__ == "__main__":
    main()
