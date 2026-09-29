"""Build viewer/index.html: a self-contained, offline browser for the 500 benchmark samples.

Added for the public release. Reads data/benchmark_500.jsonl (canonical), the
unexpanded provenance copy (for the pre-expansion plan), gate picks, and the four
generation-run plans (for archetype / severity / context metadata).

    python code/scripts/build_viewer.py
"""
from __future__ import annotations

import json
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[2]
DATA = RELEASE / "data"
TEMPLATE = Path(__file__).with_name("viewer_template.html")
OUT = RELEASE / "viewer" / "index.html"


def main() -> None:
    rows = [json.loads(l) for l in open(DATA / "benchmark_500.jsonl")]
    unexpanded = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "provenance/benchmark_500_unexpanded.jsonl")}
    picks = json.load(open(DATA / "provenance/gate_picks.json"))
    plans = {}
    for rd in ("round-a-syc", "round-b-syc", "round-a-cv", "round-b-cv"):
        attempt = rd.split("-")[1]
        for a in json.load(open(DATA / f"provenance/generation_runs/{rd}/plan.json")):
            plans[(f"FINAL-{a['final_slot_id']}", attempt)] = a

    assert len(rows) == 500 and len({r["id"] for r in rows}) == 500
    samples = []
    for r in rows:
        sid = r["id"]
        attempt = r["gate_winner_attempt"]
        a = plans[(sid, attempt)]
        er = r["evaluated_rule"]
        rule = er.get("rule", er)  # CV nests the rule; sycophancy stores it flat
        sc = r["scenario"]
        samples.append({
            "id": sid,
            "axis": r["benchmark_axis"],
            "domain": r["domain"]["name"],
            "rule": {k: rule.get(k) for k in ("rule_id", "title", "description", "example", "neighboring_boundary")},
            "archetype": {"id": a["conversation_archetype"]["id"], "name": a["conversation_archetype"]["name"],
                          "definition": a["conversation_archetype"].get("definition")},
            "severity": a.get("severity"),
            "context": a.get("scenario_context"),
            "style": r["texting_style"],
            "user_role": sc["user_role"],
            "plan": sc["scenario_plan"],
            "plan_original": unexpanded[sid]["scenario"]["scenario_plan"],
            "archetype_fit": sc.get("archetype_fit_rationale"),
            "elicitation": sc.get("rule_elicitation_rationale"),
            "turns": [{"role": t["role"], "content": t["content"]} for t in r["transcript"]["turns"]],
            "gate": {"attempt": attempt, "tied": picks[sid]["tied"],
                     "score_a": picks[sid]["score_a"], "score_b": picks[sid]["score_b"]},
            "audit": {"decision": r["terminal_audit"].get("decision"),
                      "refine_iters": r["scenario_refinement_iterations"],
                      "audit_iters": r["audit_iterations"]},
            "models": r["models"],
        })
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__SAMPLES_JSON__", payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(samples)} samples, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
