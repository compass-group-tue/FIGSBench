"""Write the exact judge requests (system + user message) for one scenario's reference
conversation, for all four stages. No model is called.

The rules stages receive the stage-1/2 score as LOCKED_SCORE; --locked-score fills it in.

    python scripts/render_judge_prompts.py FINAL-SYC-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from figsbench.judge.stages import STAGES, rules_user_prompt, score_user_prompt

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample_id")
    ap.add_argument("--locked-score", type=int, default=2)
    ap.add_argument("--out", type=Path, default=REPO / "examples" / "rendered_prompts" / "judge")
    args = ap.parse_args()

    sample = next((json.loads(l) for l in open(REPO / "data/benchmark_500.jsonl")
                   if json.loads(l)["id"] == args.sample_id), None)
    if sample is None:
        raise SystemExit(f"unknown sample id {args.sample_id}")
    transcript = [{"turn": t["turn"], "role": t["role"], "content": t["content"]}
                  for t in sample["transcript"]["turns"]]
    out = args.out / args.sample_id
    out.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(("syc-score", "cv-score", "syc-rules", "cv-rules"), 1):
        stage = STAGES[name]
        user = (score_user_prompt(stage.axis, transcript) if stage.kind == "score"
                else rules_user_prompt(transcript, args.locked_score))
        base = f"{index}_{name.replace('-', '_')}"
        (out / f"{base}.system.txt").write_text(stage.prompt_path.read_text(encoding="utf-8"))
        (out / f"{base}.user.txt").write_text(user)
    print(f"wrote 4 judge stages for {args.sample_id} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
