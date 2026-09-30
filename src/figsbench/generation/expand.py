"""Expand benchmark scenario plans into concrete, voice-neutral, turn-by-turn plans.

Reads a JSONL file of benchmark records (the format of data/benchmark_500.jsonl), sends
each record's `scenario.scenario_plan` to any OpenAI-compatible chat endpoint together
with an expansion prompt template, and writes the records back with the rewritten plan.
Standard library only.

The template is a text file with four placeholders, filled per record:
  {plan}  the current scenario_plan
  {role}  the scenario's user_role (truncated to --role-chars)
  {rule}  the evaluated rule, as "ID: title. description"
  {arch}  the scenario's archetype_fit_rationale (truncated to --arch-chars)
The default template is src/figsbench/prompts/expansion/expansion_v1.txt.

Examples
  # preview the rendered request for one record, no API call
  python -m figsbench.generation.expand data/provenance/benchmark_500_unexpanded.jsonl out.jsonl --dry-run

  # self-hosted GLM-5.3-Flash (see docker/)
  python -m figsbench.generation.expand in.jsonl out.jsonl --base-url http://localhost:8000/v1

  # any hosted OpenAI-compatible API
  EXPAND_API_KEY=... python -m figsbench.generation.expand in.jsonl out.jsonl \
      --base-url https://openrouter.ai/api/v1 --model z-ai/glm-5.3-flash --workers 16

The run is resumable: records already present in the output file are skipped, and the
output is rewritten in input order when every record is done. Records that fail after
all retries are listed at the end and left out of the output; re-run to retry them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from figsbench import PROMPTS_DIR

DEFAULT_TEMPLATE = PROMPTS_DIR / "expansion" / "expansion_v1.txt"
PLACEHOLDERS = ("{plan}", "{role}", "{rule}", "{arch}")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="input JSONL of benchmark records")
    ap.add_argument("output", type=Path, help="output JSONL (appended to while running; resumable)")
    ap.add_argument("--base-url", default=os.environ.get("EXPAND_BASE_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible base URL (env EXPAND_BASE_URL)")
    ap.add_argument("--model", default=os.environ.get("EXPAND_MODEL", "zai-org/GLM-5.3-Flash"),
                    help="model name as served by the endpoint (env EXPAND_MODEL)")
    ap.add_argument("--api-key", default=os.environ.get("EXPAND_API_KEY", ""),
                    help="bearer token, if the endpoint needs one (env EXPAND_API_KEY)")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="expansion prompt template")
    ap.add_argument("--tag", default="expansion-v1", help="value written to scenario.expansion")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=12000,
                    help="generation budget; reasoning models spend part of it thinking")
    ap.add_argument("--role-chars", type=int, default=1500, help="truncate user_role to this many characters")
    ap.add_argument("--arch-chars", type=int, default=300, help="truncate archetype rationale to this many characters")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="reject (and retry) expansions shorter than this; 0 disables")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    ap.add_argument("--ids", nargs="*", default=None, help="only expand these record ids")
    ap.add_argument("--limit", type=int, default=0, help="only expand the first N selected records")
    ap.add_argument("--keep-original", action="store_true",
                    help="store the pre-expansion plan in scenario.original_scenario_plan")
    ap.add_argument("--dry-run", action="store_true", help="print the rendered prompt for the first record and exit")
    return ap.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def format_rule(record: dict) -> str:
    rule = record.get("scenario", {}).get("evaluated_rule") or record.get("evaluated_rule") or {}
    if not isinstance(rule, dict):
        return str(rule)
    rule = rule.get("rule", rule)  # calibrated-validation records nest the rule one level down
    head = ": ".join(x for x in (rule.get("rule_id"), rule.get("title")) if x)
    desc = rule.get("description") or ""
    if head and desc:
        return f"{head}. {desc}"
    return head or desc


def render_prompt(template: str, record: dict, args: argparse.Namespace) -> str:
    sc = record["scenario"]
    fields = {
        "{plan}": sc["scenario_plan"],
        "{role}": (sc.get("user_role") or "")[: args.role_chars],
        "{rule}": format_rule(record),
        "{arch}": (sc.get("archetype_fit_rationale") or "")[: args.arch_chars],
    }
    # plain replacement, so braces elsewhere in the template or the data are left alone
    for key, value in fields.items():
        template = template.replace(key, value)
    return template


def chat(prompt: str, args: argparse.Namespace) -> str:
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        message = json.load(resp)["choices"][0]["message"]
    text = THINK_RE.sub("", message.get("content") or "").strip()
    return text


def expand_one(record: dict, template: str, args: argparse.Namespace) -> dict:
    prompt = render_prompt(template, record, args)
    last_error = "no attempts"
    for attempt in range(1, args.retries + 1):
        try:
            text = chat(prompt, args)
            if not text:
                last_error = "empty content (raise --max-tokens if the model is reasoning)"
            elif len(text) < args.min_chars:
                last_error = f"too short ({len(text)} < {args.min_chars} chars)"
            else:
                out = json.loads(json.dumps(record))
                if args.keep_original:
                    out["scenario"]["original_scenario_plan"] = record["scenario"]["scenario_plan"]
                out["scenario"]["scenario_plan"] = text
                out["scenario"]["expansion"] = args.tag
                return out
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.retries:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last_error)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = args.template.read_text(encoding="utf-8")
    missing = [p for p in PLACEHOLDERS if p not in template]
    if missing:
        print(f"warning: template has no {', '.join(missing)} placeholder(s)", file=sys.stderr)

    records = load_jsonl(args.input)
    if args.ids:
        wanted = set(args.ids)
        records = [r for r in records if r["id"] in wanted]
        unknown = wanted - {r["id"] for r in records}
        if unknown:
            print(f"warning: ids not in input: {sorted(unknown)}", file=sys.stderr)
    if args.limit:
        records = records[: args.limit]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("input has duplicate record ids; expansion results would collide")
    if not records:
        raise SystemExit("no records selected")

    if args.dry_run:
        print(render_prompt(template, records[0], args))
        return 0

    done = {r["id"] for r in load_jsonl(args.output)} if args.output.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} selected, {len(records) - len(todo)} already done, {len(todo)} to expand "
          f"via {args.model} @ {args.base_url}", flush=True)

    lock = threading.Lock()
    failed: dict[str, str] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(expand_one, r, template, args): r["id"] for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            rid = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue with the rest
                failed[rid] = str(exc)
                print(f"[{n}/{len(todo)}] FAILED {rid}: {exc}", flush=True)
                continue
            with lock, args.output.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{n}/{len(todo)}] {rid}: {len(out['scenario']['scenario_plan'])} chars", flush=True)

    if failed:
        print(f"{len(failed)} failed (re-run to retry): {sorted(failed)}", file=sys.stderr)
        return 1
    # all selected records done: rewrite the output in input order (keeps any extra ids already in the file)
    by_id = {r["id"]: r for r in load_jsonl(args.output)}
    ordered = [by_id.pop(i) for i in ids] + list(by_id.values())
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in ordered:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    print(f"done: {len(ids)} records -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
