"""Evaluate a model: python -m figsbench.evaluation --samples data/benchmark_500.jsonl --model ID ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EvalConfig, load_config
from .runner import evaluate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--model", dest="model_under_test", default=None)
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--served-model", default=None)
    parser.add_argument("--endpoint-api-key", default=None)
    parser.add_argument("--assistant-prompt", default=None)
    parser.add_argument("--user-model", default=None,
                        help="local-deepseek-v4-flash (paper, default) | api (= deepseek/deepseek-v4-flash "
                             "on OpenRouter) | any OpenRouter id")
    parser.add_argument("--judge", choices=("skip", "local", "api"), default=None,
                        help="local = paper's GLM-5.3-Flash server (default); api = OpenRouter")
    parser.add_argument("--judge-model", default=None,
                        help="OpenRouter judge for --judge api (default google/gemini-3.8-flash)")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--service-tier", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--assistant-max-tokens", type=int, default=None,
                        help="per-turn cap for the model under test; 0 = uncapped (omit max_tokens)")
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=None,
        help="optional reproducibility hint passed to model providers",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help="replace selected transcripts instead of safely resuming them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data: dict = {}
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8"))
    overrides = {
        key: value
        for key in (
            "samples", "model_under_test", "endpoint_url", "served_model",
            "endpoint_api_key", "assistant_prompt", "user_model", "judge", "judge_model",
            "reasoning_effort", "service_tier", "output_dir", "sample_ids",
            "limit", "assistant_max_tokens", "rollout_seed", "workers",
            "retries", "timeout", "force",
        )
        if (value := getattr(args, key)) is not None
    }
    data.update(overrides)
    if data.get("assistant_max_tokens") == 0:
        data["assistant_max_tokens"] = None
    if "samples" not in data:
        raise SystemExit("--samples (or a config file with samples) is required")
    report = evaluate(EvalConfig(**data))
    print(json.dumps(
        {
            "output_dir": str(report.output_dir),
            "completed": len(report.completed),
            "failed": report.failed,
            "judge_run_id": report.judge_run_id,
            "judge_counts": report.judge_counts,
            "judge_status": report.judge_status,
            "judge_error": report.judge_error,
        },
        indent=2,
    ))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
