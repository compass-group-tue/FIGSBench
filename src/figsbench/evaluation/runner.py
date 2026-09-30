"""One-call evaluation: roll out every scenario against the model under test, judge the
conversations with the four frozen stages, and compute the benchmark score."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from figsbench import PROMPTS_DIR
from figsbench.client import build_client, build_eval_client, load_dotenv
from figsbench.generation.prompts import user_simulator_prompt

from .config import PAPER_USER_MODEL, EvalConfig, EvalReport
from .prices import cost_usd

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_assistant_prompt(spec: str) -> str:
    """A prompt name from prompts/assistant/ ("baseline", "factual-v2", ...) or a file path."""
    name = spec.strip().lower().replace("-", "_")
    named = PROMPTS_DIR / "assistant" / f"{name}.txt"
    if named.is_file():
        return named.read_text(encoding="utf-8")
    path = Path(spec).expanduser()
    if not path.is_file():
        choices = ", ".join(sorted(p.stem.replace("_", "-")
                                   for p in (PROMPTS_DIR / "assistant").glob("*.txt")))
        raise ValueError(f"unknown assistant prompt {spec!r}: use one of {choices}, "
                         "or a prompt file path")
    return path.read_text(encoding="utf-8")


def _load_samples(source: Path) -> list[dict[str, Any]]:
    """Load either a canonical JSONL file or a run directory."""
    if source.is_dir():
        path = source / "samples.jsonl"
    elif source.is_file():
        path = source
    else:
        raise FileNotFoundError(f"samples path does not exist: {source}")
    if not path.is_file():
        raise FileNotFoundError(f"samples JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object in {path}:{line_number}")
        records.append(value)
    return records


def _rollout(
    *,
    sample: dict[str, Any],
    assistant_client: Any,
    user_client: Any,
    assistant_system_prompt: str,
    assistant_max_tokens: int | None = 4000,
    rollout_seed: int | None = None,
) -> dict[str, Any]:
    scenario = sample["scenario"]
    texting_style = sample.get("texting_style") or {}
    assistant_client = copy.copy(assistant_client)
    assistant_client.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                              "cached_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    assistant_client._usage_lock = threading.Lock()
    user_client = copy.copy(user_client)
    user_client.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                         "cached_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    user_client._usage_lock = threading.Lock()
    history: list[dict[str, str]] = []

    def call_seed(role: str, turn: int) -> int | None:
        if rollout_seed is None:
            return None
        digest = hashlib.sha256(
            f"{rollout_seed}:{sample['id']}:{role}:{turn}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    for user_turn in range(1, 6):
        user_options: dict[str, Any] = {}
        user_seed = call_seed("user", user_turn)
        if user_seed is not None:
            user_options["seed"] = user_seed
        user_message = user_client.complete(
            system=user_simulator_prompt(scenario, texting_style, user_turn, history),
            messages=[
                {
                    "role": "user",
                    "content": "Write the next in-character user message only.",
                }
            ],
            temperature=0.85,
            max_tokens=1200,
            **user_options,
        ).strip()
        if not user_message:
            raise ValueError(f"{sample['id']}: user model returned empty text")
        history.append({"role": "user", "content": user_message})
        assistant_options: dict[str, Any] = {}
        assistant_seed = call_seed("assistant", user_turn)
        if assistant_seed is not None:
            assistant_options["seed"] = assistant_seed
        assistant_message = assistant_client.complete(
            system=assistant_system_prompt,
            messages=[
                {"role": item["role"], "content": item["content"]}
                for item in history
            ],
            temperature=0.25,
            max_tokens=assistant_max_tokens,
            # Final user turn is never re-read: skip its cache breakpoint
            # to avoid a pure-waste cache write (writes bill ~1.25x input).
            cache_tail=(user_turn < 5),
            **assistant_options,
        ).strip()
        if not assistant_message:
            raise ValueError(f"{sample['id']}: model under test returned empty text")
        history.append({"role": "assistant", "content": assistant_message})
    assistant_usage = dict(assistant_client.usage)
    user_usage = dict(user_client.usage)
    assistant_cost = cost_usd(
        assistant_client.model,
        assistant_usage,
        getattr(assistant_client, "service_tier", None),
    )
    user_cost = cost_usd(user_client.model, user_usage)
    return {
        "schema_version": 2,
        "scenario_id": sample["id"],
        "benchmark_axis": scenario.get("benchmark_axis"),
        "assistant_model": assistant_client.model,
        "user_model": user_client.model,
        "usage": {"assistant": assistant_usage, "user": user_usage},
        "cost_usd": {"assistant": assistant_cost, "user": user_cost},
        "turns": [
            {"turn": index + 1, "role": item["role"], "content": item["content"]}
            for index, item in enumerate(history)
        ],
    }


def _log_rollout_cost(sample_id: str, transcript: dict[str, Any]) -> None:
    cost = transcript.get("cost_usd") or {}
    usage = transcript.get("usage") or {}
    a = usage.get("assistant", {})
    price = cost.get("assistant")
    print(
        f"EVAL {sample_id} done "
        f"asst_cost=${price if price is not None else 'n/a'} "
        f"(in={a.get('prompt_tokens', 0)} cached={a.get('cached_tokens', 0)} "
        f"out={a.get('completion_tokens', 0)} think={a.get('reasoning_tokens', 0)})",
        flush=True,
    )


def _sum_costs(
    out_dir: Path, sample_ids: set[str] | None = None
) -> dict[str, Any]:
    total: dict[str, Any] = {"assistant": 0.0, "user": 0.0, "unknown_price": 0}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
             "reasoning_tokens": 0, "calls": 0}
    n = 0
    for path in sorted((out_dir / "transcripts").glob("*.json")):
        if sample_ids is not None and path.stem not in sample_ids:
            continue
        try:
            t = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        c = t.get("cost_usd") or {}
        u = (t.get("usage") or {}).get("assistant", {})
        if c.get("assistant") is None:
            total["unknown_price"] += 1
        else:
            total["assistant"] += c["assistant"]
        if c.get("user") is not None:
            total["user"] += c["user"]
        for k in usage:
            usage[k] += int(u.get(k, 0) or 0)
        n += 1
    total["assistant"] = round(total["assistant"], 4)
    total["user"] = round(total["user"], 4)
    return {"n_transcripts": n, "cost_usd": total, "assistant_usage": usage}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sample_payload(sample: dict[str, Any]) -> dict[str, Any]:
    """Return immutable benchmark content, excluding earlier run products."""
    return {
        key: value
        for key, value in sample.items()
        if key not in {"transcript", "models"}
    }


def _client_identity(client: Any, requested_model: str) -> dict[str, Any]:
    """Capture behavior-affecting client fields without persisting credentials."""
    return {
        "requested_model": requested_model,
        "client_model": getattr(client, "model", None),
        "served_model": getattr(client, "served_model_name", None),
        "base_url": getattr(client, "base_url", None),
        "reasoning_effort": getattr(client, "reasoning_effort", None),
        "provider": getattr(client, "provider", None),
        "service_tier": getattr(client, "service_tier", None),
    }


def _behavior_spec(
    *,
    config: EvalConfig,
    assistant_client: Any,
    user_client: Any,
    assistant_system_prompt: str,
) -> dict[str, Any]:
    return {
        "fingerprint_version": 1,
        "assistant": _client_identity(
            assistant_client, config.model_under_test
        ),
        "user": _client_identity(user_client, config.user_model),
        "assistant_prompt_sha256": hashlib.sha256(
            assistant_system_prompt.encode("utf-8")
        ).hexdigest(),
        "rollout": {
            "turns_per_role": 5,
            "assistant_temperature": 0.25,
            "assistant_max_tokens": config.assistant_max_tokens,
            "user_temperature": 0.85,
            "user_max_tokens": 1200,
            "rollout_seed": config.rollout_seed,
        },
    }


def _evaluation_metadata(
    *,
    sample: dict[str, Any],
    behavior_spec: dict[str, Any],
) -> dict[str, Any]:
    sample_fingerprint = _sha256_json(_sample_payload(sample))
    behavior_fingerprint = _sha256_json(behavior_spec)
    rollout_fingerprint = _sha256_json(
        {
            "sample_fingerprint": sample_fingerprint,
            "behavior_fingerprint": behavior_fingerprint,
        }
    )
    assistant = behavior_spec["assistant"]
    user = behavior_spec["user"]
    return {
        "fingerprint_version": behavior_spec["fingerprint_version"],
        "sample_fingerprint": sample_fingerprint,
        "behavior_fingerprint": behavior_fingerprint,
        "rollout_fingerprint": rollout_fingerprint,
        "assistant": {
            key: assistant.get(key)
            for key in (
                "requested_model",
                "client_model",
                "served_model",
                "reasoning_effort",
                "service_tier",
            )
        },
        "user": {
            key: user.get(key)
            for key in ("requested_model", "client_model", "served_model")
        },
        "assistant_prompt_sha256": behavior_spec["assistant_prompt_sha256"],
        "settings": behavior_spec["rollout"],
    }


def _load_reusable_transcript(
    path: Path,
    *,
    sample: dict[str, Any],
    expected_metadata: dict[str, Any],
    assistant_model: str,
    user_model: str,
) -> dict[str, Any]:
    """Load and strictly validate a transcript before treating it as resumed."""
    try:
        transcript = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing transcript {path}: {exc}") from exc
    if not isinstance(transcript, dict):
        raise ValueError(f"existing transcript {path} is not a JSON object")
    if transcript.get("schema_version") != 2:
        raise ValueError(
            f"existing transcript {path} has unsupported schema_version="
            f"{transcript.get('schema_version')!r}"
        )
    if transcript.get("scenario_id") != sample.get("id"):
        raise ValueError(
            f"existing transcript {path} has scenario_id="
            f"{transcript.get('scenario_id')!r}"
        )
    actual_metadata = transcript.get("evaluation")
    if not isinstance(actual_metadata, dict):
        raise ValueError(
            f"existing transcript {path} has no evaluation fingerprint"
        )
    if actual_metadata.get("rollout_fingerprint") != expected_metadata.get(
        "rollout_fingerprint"
    ):
        raise ValueError(
            f"existing transcript {path} was produced with different sample, "
            "model, user, prompt, or rollout settings"
        )
    transcript_fingerprint = actual_metadata.get("transcript_sha256")
    transcript_payload = {
        key: value for key, value in transcript.items() if key != "evaluation"
    }
    if (
        not isinstance(transcript_fingerprint, str)
        or transcript_fingerprint != _sha256_json(transcript_payload)
    ):
        raise ValueError(
            f"existing transcript {path} content does not match its fingerprint"
        )
    if transcript.get("assistant_model") != assistant_model:
        raise ValueError(
            f"existing transcript {path} has assistant_model="
            f"{transcript.get('assistant_model')!r}, expected {assistant_model!r}"
        )
    if transcript.get("user_model") != user_model:
        raise ValueError(
            f"existing transcript {path} has user_model="
            f"{transcript.get('user_model')!r}, expected {user_model!r}"
        )
    expected_axis = sample.get("scenario", {}).get("benchmark_axis")
    if transcript.get("benchmark_axis") != expected_axis:
        raise ValueError(
            f"existing transcript {path} has benchmark_axis="
            f"{transcript.get('benchmark_axis')!r}, expected {expected_axis!r}"
        )
    turns = transcript.get("turns")
    if not isinstance(turns, list) or len(turns) != 10:
        raise ValueError(f"existing transcript {path} must contain exactly 10 turns")
    for index, turn in enumerate(turns, start=1):
        expected_role = "user" if index % 2 else "assistant"
        if (
            not isinstance(turn, dict)
            or turn.get("turn") != index
            or turn.get("role") != expected_role
            or not isinstance(turn.get("content"), str)
            or not turn["content"].strip()
        ):
            raise ValueError(
                f"existing transcript {path} has an invalid turn at position {index}"
            )
    return transcript


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _duplicate_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _safe_run_label(value: str) -> str:
    label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")
    return label[:80] or "evaluation"


def _collect_judgments(
    *,
    out_dir: Path,
    judge_run_id: str,
    sample_ids: list[str],
) -> list[dict[str, Any]]:
    """Load the merged judgments of a complete judge run and point latest.json at it."""
    source = out_dir / "judgments" / judge_run_id
    judgments: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        source_path = source / "results" / f"{sample_id}.json"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"complete judge run is missing result for {sample_id}: {source_path}"
            )
        value = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"judge result is not a JSON object: {source_path}")
        judgments.append(value)
    _write_json_atomic(
        out_dir / "judgments" / "latest.json",
        {
            "judge_run_id": judge_run_id,
            "path": str(Path(judge_run_id) / "manifest.json"),
        },
    )
    return judgments


def evaluate(config: EvalConfig) -> EvalReport:
    """Run an evaluation sweep; return paths plus per-sample outcomes."""
    config.validate()
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    samples = (
        list(config.samples)
        if isinstance(config.samples, list)
        else _load_samples(Path(config.samples))
    )
    if config.sample_ids:
        duplicate_requested = _duplicate_ids(config.sample_ids)
        if duplicate_requested:
            raise ValueError(
                f"duplicate selected sample ids: {duplicate_requested}"
            )
        wanted = set(config.sample_ids)
        samples = [s for s in samples if s.get("id") in wanted]
        missing = wanted - {
            str(s.get("id")) for s in samples if isinstance(s.get("id"), str)
        }
        if missing:
            raise ValueError(f"unknown sample ids: {sorted(missing)}")
    if config.limit:
        samples = samples[: config.limit]
    if not samples:
        raise ValueError("no samples selected")
    selected_ids: list[str] = []
    for sample in samples:
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("selected sample has a missing or invalid id")
        selected_ids.append(sample_id)
    duplicates = _duplicate_ids(selected_ids)
    if duplicates:
        raise ValueError(f"duplicate selected sample ids: {duplicates}")
    from .ground_truth import assert_final_samples

    assert_final_samples(samples)

    out_dir = Path(config.output_dir)
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    assistant_system_prompt = resolve_assistant_prompt(config.assistant_prompt)
    assistant_client = build_eval_client(
        model=config.model_under_test,
        api_key=api_key,
        endpoint_url=config.endpoint_url,
        served_model=config.served_model,
        endpoint_api_key=config.endpoint_api_key,
        reasoning_effort=config.reasoning_effort,
        timeout=config.timeout,
        retries=config.retries,
        service_tier=config.service_tier,
    )
    user_client = build_client(
        config.user_model,
        api_key=api_key,
        timeout=config.timeout,
        retries=config.retries,
    )
    behavior_spec = _behavior_spec(
        config=config,
        assistant_client=assistant_client,
        user_client=user_client,
        assistant_system_prompt=assistant_system_prompt,
    )
    expected_metadata = {
        sample["id"]: _evaluation_metadata(
            sample=sample,
            behavior_spec=behavior_spec,
        )
        for sample in samples
    }
    behavior_fingerprint = _sha256_json(behavior_spec)
    run_fingerprint = _sha256_json(
        {
            "behavior_fingerprint": behavior_fingerprint,
            "samples": [
                expected_metadata[sample_id]["rollout_fingerprint"]
                for sample_id in selected_ids
            ],
        }
    )

    pending: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    for sample in samples:
        path = transcripts_dir / f"{sample['id']}.json"
        if config.force or not path.exists():
            pending.append(sample)
            continue
        if not path.is_file():
            raise ValueError(
                f"existing transcript path is not a file: {path}; "
                "remove it or choose another output directory"
            )
        try:
            _load_reusable_transcript(
                path,
                sample=sample,
                expected_metadata=expected_metadata[sample["id"]],
                assistant_model=assistant_client.model,
                user_model=user_client.model,
            )
        except ValueError as exc:
            raise ValueError(
                f"{exc}. Refusing unsafe resume; pass force=True or --force "
                "to replace selected transcripts"
            ) from exc
        completed_ids.add(sample["id"])

    _write_text_atomic(out_dir / "config.json", config.to_json() + "\n")
    report = EvalReport(
        output_dir=out_dir,
        judge_status="skipped" if config.judge == "skip" else None,
    )

    def axis_of(sample: dict[str, Any]) -> str:
        return str(
            sample.get("benchmark_axis")
            or sample.get("scenario", {}).get("benchmark_axis")
            or "sycophancy"
        )

    for axis in ("sycophancy", "calibrated_validation"):
        group = [s for s in pending if axis_of(s) == axis]
        if not group:
            continue
        with ThreadPoolExecutor(
            max_workers=min(config.workers, len(group))
        ) as executor:
            futures = {
                executor.submit(
                    _rollout,
                    sample=sample,
                    assistant_client=assistant_client,
                    user_client=user_client,
                    assistant_system_prompt=assistant_system_prompt,
                    assistant_max_tokens=config.assistant_max_tokens,
                    rollout_seed=config.rollout_seed,
                ): sample
                for sample in group
            }
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    transcript = future.result()
                except Exception as exc:
                    report.failed[sample["id"]] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"EVAL {sample['id']} FAILED: {exc}", flush=True)
                    continue
                metadata = dict(expected_metadata[sample["id"]])
                metadata["transcript_sha256"] = _sha256_json(transcript)
                transcript["evaluation"] = metadata
                _write_json_atomic(
                    transcripts_dir / f"{sample['id']}.json", transcript
                )
                completed_ids.add(sample["id"])
                _log_rollout_cost(sample["id"], transcript)

    report.completed = [
        sample_id for sample_id in selected_ids if sample_id in completed_ids
    ]

    judged_samples: list[dict[str, Any]] = []
    for sample in samples:
        if sample["id"] not in completed_ids:
            continue
        path = transcripts_dir / f"{sample['id']}.json"
        entry = deepcopy(sample)
        entry["transcript"] = json.loads(path.read_text(encoding="utf-8"))
        entry["models"] = {
            "assistant": assistant_client.model,
            "user": user_client.model,
        }
        judged_samples.append(entry)
    _write_text_atomic(
        out_dir / "samples.jsonl",
        "".join(
            json.dumps(entry, ensure_ascii=False) + "\n"
            for entry in judged_samples
        ),
    )
    eval_manifest: dict[str, Any] = {
        "status": "complete" if not report.failed else "incomplete",
        "rollout_status": "complete" if not report.failed else "incomplete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "behavior_fingerprint": behavior_fingerprint,
        "run_fingerprint": run_fingerprint,
        "requested_samples": len(samples),
        "completed_samples": len(judged_samples),
    }
    _write_json_atomic(out_dir / "manifest.json", eval_manifest)

    # Never leave an old score beside a skipped or incomplete new evaluation.
    for stale_product in ("metrics.json", "per_sample_scores.jsonl"):
        (out_dir / stale_product).unlink(missing_ok=True)
    (out_dir / "judgments" / "latest.json").unlink(missing_ok=True)

    if config.judge != "skip":
        if not judged_samples:
            report.judge_status = "not_run"
            report.judge_error = "no completed transcripts to judge"
        else:
            try:
                from figsbench.judge import API_JUDGE_MODEL, PAPER_JUDGE_MODEL, run_judge

                judge_model = (
                    (config.judge_model or API_JUDGE_MODEL)
                    if config.judge == "api"
                    else PAPER_JUDGE_MODEL
                )
                paper_judge = judge_model == PAPER_JUDGE_MODEL
                paper_user = config.user_model == PAPER_USER_MODEL
                protocol = {
                    "judge_backend": config.judge,
                    "judge_model": judge_model,
                    "paper_judge": paper_judge,
                    "user_model": config.user_model,
                    "paper_user_simulator": paper_user,
                    "paper_protocol": paper_judge and paper_user,
                }
                eval_manifest.update(protocol)
                judge_fingerprint = _sha256_json(
                    {
                        "run_fingerprint": run_fingerprint,
                        "judge_backend": config.judge,
                        "judge_model": judge_model,
                    }
                )
                judge_run_id = (
                    f"{_safe_run_label(out_dir.name)}-judged-"
                    f"{config.judge}-{judge_fingerprint[:24]}"
                )
                report.judge_run_id = judge_run_id
                judge_manifest = run_judge(
                    source_run=out_dir,
                    out_root=out_dir / "judgments",
                    judge_run_id=judge_run_id,
                    judge_model=judge_model,
                    api_key=api_key,
                    workers=min(24, len(judged_samples)),
                )
                report.judge_counts = judge_manifest.get("counts")
                report.judge_status = str(
                    judge_manifest.get("status") or "unknown"
                )
                if report.judge_status == "complete" and isinstance(
                    report.judge_counts, dict
                ):
                    success_count = int(
                        report.judge_counts.get("success", 0) or 0
                    )
                    other_count = sum(
                        int(value or 0)
                        for key, value in report.judge_counts.items()
                        if key != "success"
                    )
                    if success_count != len(judged_samples) or other_count:
                        report.judge_status = "partial"
                if report.judge_status == "complete" and not report.failed:
                    judgments = _collect_judgments(
                        out_dir=out_dir,
                        judge_run_id=judge_run_id,
                        sample_ids=[entry["id"] for entry in judged_samples],
                    )
                    from figsbench.metrics import summarize

                    metrics, per_sample_scores = summarize(
                        judged_samples,
                        judgments,
                    )
                    # Scores from any judge other than the paper's local GLM-5.3-Flash
                    # are a different protocol and must not be compared with the paper.
                    metrics.update(protocol)
                    _write_json_atomic(out_dir / "metrics.json", metrics)
                    _write_text_atomic(
                        out_dir / "per_sample_scores.jsonl",
                        "".join(
                            json.dumps(row, ensure_ascii=False) + "\n"
                            for row in per_sample_scores
                        ),
                    )
            except Exception as exc:
                report.judge_status = "failed"
                report.judge_error = f"{type(exc).__name__}: {exc}"
                print(f"EVAL JUDGE FAILED: {exc}", flush=True)

    eval_manifest["judge_run_id"] = report.judge_run_id
    eval_manifest["judge_status"] = report.judge_status
    eval_manifest["judge_counts"] = report.judge_counts
    if report.judge_error:
        eval_manifest["judge_error"] = report.judge_error
    eval_manifest["status"] = "complete" if report.ok else "incomplete"
    _write_json_atomic(out_dir / "manifest.json", eval_manifest)

    totals = _sum_costs(out_dir, set(completed_ids))
    print(f"EVAL TOTAL {out_dir.name}: {totals}", flush=True)
    _write_json_atomic(
        out_dir / "summary.json",
        {
            "completed": sorted(report.completed),
            "failed": report.failed,
            "judge_run_id": report.judge_run_id,
            "judge_counts": report.judge_counts,
            "judge_status": report.judge_status,
            "judge_error": report.judge_error,
            "behavior_fingerprint": behavior_fingerprint,
            "run_fingerprint": run_fingerprint,
            "cost_totals": totals,
        },
    )
    return report
