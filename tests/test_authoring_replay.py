"""Replay one recorded benchmark scenario through the whole authoring pipeline.

A fake model answers every stage with what the paper's run produced for FINAL-SYC-001 and
FINAL-CV-001 (data/provenance/). The pipeline must finish and reproduce the stored scenario,
refinement history and audit decisions.
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pytest

from figsbench.generation.pipeline import run_authoring

REPO = Path(__file__).resolve().parents[1]
PROV = REPO / "data" / "provenance"
EXTRA = ("rule_elicitation_rationale", "archetype_fit_rationale", "conversation_archetype_id")


def _record(sample_id: str) -> dict:
    for line in open(PROV / "benchmark_500_unexpanded.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row["id"] == sample_id:
            return row
    raise KeyError(sample_id)


def _queues(sample: dict) -> dict[str, list[str]]:
    final = sample["scenario"]
    extra = {k: final[k] for k in EXTRA if k in final}
    cv = sample["benchmark_axis"] == "calibrated_validation"
    q = {k: [] for k in ("generator", "refiner", "revision", "user", "assistant", "audit")}
    scen = lambda d: json.dumps({**extra, "user_role": d["user_role"], "scenario_plan": d["scenario_plan"]})
    hist = sample["scenario_refinement_history"]
    q["generator"].append(scen(hist[0]["scenario_before"]))
    for rec in hist:
        fb = {k: rec[k] for k in ("verdict", "strengths", "issues", "revision_brief")}
        if cv:
            fb["archetype_fit_assessment"] = "fits"
            fb["rule_elicitation_assessment"] = "elicits"
        q["refiner"].append(json.dumps(fb))
        q["revision"].append(scen(rec["scenario_after"]))

    def talk(turns):
        for t in turns:
            q["user" if t["role"] == "user" else "assistant"].append(t["content"])

    talk(sample["original_transcript"]["turns"])
    for audit in sample["audit_history"]:
        q["audit"].append(json.dumps({k: v for k, v in audit.items() if "fingerprint" not in k}))
        if audit["decision"] == "revise_and_reroll":
            talk(sample["transcript"]["turns"])
    q["audit"].append(json.dumps(sample["terminal_audit"]))
    return q


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("sample_id,run", [("FINAL-SYC-001", "sycophancy"),
                                           ("FINAL-CV-001", "calibrated_validation")])
def test_pipeline_reproduces_a_recorded_scenario(sample_id, run, tmp_path, monkeypatch):
    sample = _record(sample_id)
    plan = json.load(open(PROV / "generation_runs" / f"attempt-{sample['gate_winner_attempt']}" / run / "plan.json"))
    assignment = next(a for a in plan if a["id"] == sample_id)
    q = _queues(sample)
    stages = {(0.9, True): "generator", (0.25, True): "refiner", (0.65, True): "revision",
              (0.85, False): "user", (0.25, False): "assistant", (0.15, True): "audit"}

    def fake(request, timeout=None):
        payload = json.loads(request.data)
        content = q[stages[(payload["temperature"], "response_format" in payload)]].pop(0)
        return _Resp(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    manifest = run_authoring(
        api_key="test", axis_name=sample["benchmark_axis"], plan=[assignment],
        run_dir=tmp_path / "run", planning_seed=20260915,
        seed_corpus_path=REPO / "src/figsbench/data/example_seeds/seeds.jsonl",
        generator_model="g", refiner_model="g", auditor_model="g", assistant_model="a",
        user_model="local-deepseek-v4-flash", workers=1)
    assert manifest["status"] == "complete", manifest["failures"]
    assert all(not v for v in q.values())
    out = json.loads((tmp_path / "run" / "samples" / sample_id / "sample.json").read_text())
    assert out["scenario"] == sample["scenario"]
    assert out["terminal_audit"] == sample["terminal_audit"]
    assert [a["decision"] for a in out["audit_history"]] == [a["decision"] for a in sample["audit_history"]]
