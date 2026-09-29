"""Configuration for easy benchmark evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# The paper's user simulator: local vLLM serving deepseek-ai/DeepSeek-V4-Flash @ 60d8d707
# (the 0423 release; not DeepSeek-V4-Flash-0731).
PAPER_USER_MODEL = "local-deepseek-v4-flash"
# user_model="api" resolves to the same 0423 weights on OpenRouter (Alibaba fp8 route).
API_USER_MODEL = "deepseek/deepseek-v4-flash"


@dataclass
class EvalConfig:
    """Everything needed for one evaluation sweep.

    samples: benchmark JSONL, run-dir containing samples.jsonl, or a list of
        sample dicts.
    model_under_test: OpenRouter model id, `local-*` alias, or any name when
        endpoint_url is given.
    endpoint_url: custom OpenAI-compatible base URL (e.g. self-hosted vLLM).
        Requires served_model when set.
    served_model: model name served at endpoint_url.
    endpoint_api_key: optional bearer token for endpoint_url.
    assistant_prompt: "baseline" | "extreme-cold" | "factual-v2" | path to a prompt file.
    user_model: user-simulator model id. `local-deepseek-v4-flash` (default) is the
        paper's self-hosted simulator; "api" means `deepseek/deepseek-v4-flash` on
        OpenRouter (same 0423 weights); any other OpenRouter id is also accepted.
    judge: "skip" | "local" (default) | "api".
        "local" is the paper's frozen GLM-5.3-Flash judge on a self-hosted server.
        "api" judges through OpenRouter with the same frozen prompts.
    judge_model: OpenRouter judge model for judge="api" (default
        google/gemini-3.8-flash). Not allowed with "local" or "skip".
    reasoning_effort: passed through to OpenRouter for the model under test.
    service_tier: OpenRouter service tier (e.g. "flex" for 50% off OpenAI
        routes). Defaults to "flex" for gpt-6-astra, None otherwise.
    output_dir: where transcripts, samples.jsonl, and summary go.
    sample_ids / limit: subset selection. workers / retries / timeout: runtime.
    rollout_seed: optional provider seed hint, deterministically separated by
        sample, role, and turn. Providers may ignore it.
    force: explicitly replace selected transcripts instead of safe resume.
    """

    samples: str | Path | list[dict[str, Any]]
    model_under_test: str = "openai/gpt-5.6-luna"
    endpoint_url: str | None = None
    served_model: str | None = None
    endpoint_api_key: str = ""
    assistant_prompt: str = "baseline"
    user_model: str = "local-deepseek-v4-flash"
    judge: str = "local"
    judge_model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    output_dir: str | Path = "evals/eval-run"
    sample_ids: list[str] | None = None
    limit: int = 0
    assistant_max_tokens: int | None = 4000
    workers: int = 5
    retries: int = 5
    timeout: float = 180.0
    rollout_seed: int | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if self.user_model.strip().lower() == "api":
            self.user_model = API_USER_MODEL

    def validate(self) -> None:
        if self.judge not in ("skip", "local", "api"):
            raise ValueError("judge must be skip, local, or api")
        if self.judge_model and self.judge != "api":
            raise ValueError(
                "judge_model is only used with judge='api'; judge='local' is always "
                "the frozen GLM-5.3-Flash judge"
            )
        if self.endpoint_url and not self.served_model:
            raise ValueError("served_model is required when endpoint_url is set")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if isinstance(self.rollout_seed, bool) or (
            self.rollout_seed is not None
            and not isinstance(self.rollout_seed, int)
        ):
            raise ValueError("rollout_seed must be an integer or null")

    def to_json(self) -> str:
        data = asdict(self)
        # Credentials and destructive/runtime-only controls must never be
        # persisted in a reusable run configuration.
        data.pop("endpoint_api_key", None)
        data.pop("force", None)
        data["samples"] = (
            str(self.samples)
            if isinstance(self.samples, Path)
            else self.samples
        )
        data["output_dir"] = str(self.output_dir)
        return json.dumps(data, indent=2)


@dataclass
class EvalReport:
    output_dir: Path
    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    judge_run_id: str | None = None
    judge_counts: dict[str, Any] | None = None
    judge_status: str | None = None
    judge_error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.failed and self.judge_status in (
            None,
            "skipped",
            "complete",
        )


def load_config(path: str | Path) -> EvalConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalConfig(**data)
