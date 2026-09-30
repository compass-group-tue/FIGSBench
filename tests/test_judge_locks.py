"""The four judge prompts are frozen: a changed prompt must stop the judge before any call."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from figsbench.judge import stages


@pytest.mark.parametrize("name", ["syc-score", "cv-score", "syc-rules", "cv-rules"])
def test_shipped_prompts_match_the_lock(name: str) -> None:
    stage = stages.STAGES[name]
    assert stages.sha256_text(stage.prompt_path.read_text(encoding="utf-8")) == stage.lock["sha256"]


def test_tampered_prompt_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    for path in stages.JUDGE_DIR.iterdir():
        (judge_dir / path.name).write_bytes(path.read_bytes())
    (judge_dir / "syc_score_v1.txt").write_text("a different prompt", encoding="utf-8")
    monkeypatch.setattr(stages, "JUDGE_DIR", judge_dir)
    monkeypatch.setattr(stages, "LOCK_PATH", judge_dir / "lock.json")
    run = tmp_path / "run"
    run.mkdir()
    turns = [{"turn": i, "role": "user" if i % 2 else "assistant", "content": f"t{i}"}
             for i in range(1, 11)]
    (run / "samples.jsonl").write_text(json.dumps({"id": "S1", "transcript": {"turns": turns}}) + "\n")
    with pytest.raises(ValueError, match="frozen syc-score judge prompt hash mismatch"):
        stages.run_stage(stage=stages.STAGES["syc-score"], source_run=run, out_root=tmp_path,
                         run_id="x", api_key="", model="local-glm-5.3-flash", workers=1)
