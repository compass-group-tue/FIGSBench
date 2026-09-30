from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from figsbench import client as pipeline_client
from figsbench.client import CustomVLLMClient
from figsbench.evaluation import EvalConfig, EvalReport
from figsbench.evaluation import __main__ as cli
from figsbench.evaluation import runner


def make_sample(identifier: str = "S1") -> dict[str, Any]:
    return {
        "id": identifier,
        "benchmark_axis": "sycophancy",
        "scenario": {
            "benchmark_axis": "sycophancy",
            "setting": "A test conversation.",
            "user_role": "A user testing an assistant.",
            "scenario_plan": "Ask for advice and follow up naturally.",
        },
        "texting_style": {"description": "Clear, concise prose."},
        "evaluated_rule": {"rule_id": "S1.a"},
    }


class FakeClient:
    def __init__(
        self,
        model: str,
        *,
        served_model_name: str | None = None,
        service_tier: str | None = None,
    ) -> None:
        self.model = model
        self.served_model_name = served_model_name
        self.service_tier = service_tier
        self.reasoning_effort = None
        self.provider = None
        self.usage = pipeline_client.new_usage()
        self._usage_lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"{self.model} response {len(self.calls)}"


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[FakeClient]]:
    from figsbench.evaluation import ground_truth

    monkeypatch.setattr(ground_truth, "assert_final_samples", lambda samples: None)
    monkeypatch.setattr(
        runner, "resolve_assistant_prompt", lambda prompt: f"resolved:{prompt}"
    )
    built: dict[str, list[FakeClient]] = {"assistant": [], "user": []}

    def build_assistant(**kwargs: Any) -> FakeClient:
        model = kwargs.get("served_model") or kwargs["model"]
        value = FakeClient(
            model,
            served_model_name=kwargs.get("served_model"),
            service_tier=kwargs.get("service_tier"),
        )
        value.reasoning_effort = kwargs.get("reasoning_effort")
        built["assistant"].append(value)
        return value

    def build_user(model: str, **kwargs: Any) -> FakeClient:
        value = FakeClient(model)
        built["user"].append(value)
        return value

    monkeypatch.setattr(runner, "build_eval_client", build_assistant)
    monkeypatch.setattr(runner, "build_client", build_user)
    return built


def write_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )


def test_jsonl_evaluation_is_seeded_atomic_and_safely_resumable(
    tmp_path: Path,
    fake_runtime: dict[str, list[FakeClient]],
) -> None:
    samples_path = tmp_path / "benchmark.jsonl"
    output = tmp_path / "run"
    write_samples(samples_path, [make_sample()])
    config = EvalConfig(
        samples=samples_path,
        model_under_test="model-a",
        endpoint_api_key="top-secret",
        assistant_prompt="baseline",
        user_model="user-a",
        judge="skip",
        output_dir=output,
        rollout_seed=123,
        workers=1,
    )

    first = runner.evaluate(config)
    assert first.ok
    assert first.completed == ["S1"]
    assert len(fake_runtime["assistant"][0].calls) == 5
    assert len(fake_runtime["user"][0].calls) == 5
    assistant_seeds = [call["seed"] for call in fake_runtime["assistant"][0].calls]
    user_seeds = [call["seed"] for call in fake_runtime["user"][0].calls]
    assert len(set(assistant_seeds + user_seeds)) == 10
    assert fake_runtime["assistant"][0].calls[-1]["cache_tail"] is False

    transcript = json.loads((output / "transcripts" / "S1.json").read_text())
    assert transcript["evaluation"]["settings"]["rollout_seed"] == 123
    assert len(transcript["evaluation"]["rollout_fingerprint"]) == 64
    persisted_config = (output / "config.json").read_text()
    assert "endpoint_api_key" not in persisted_config
    assert "top-secret" not in persisted_config
    assert not list(output.rglob("*.tmp"))

    second = runner.evaluate(config)
    assert second.ok
    assert second.completed == ["S1"]
    assert len(fake_runtime["assistant"][1].calls) == 0
    assert len(fake_runtime["user"][1].calls) == 0

    changed = EvalConfig(
        **{
            **config.__dict__,
            "model_under_test": "model-b",
        }
    )
    with pytest.raises(ValueError, match="Refusing unsafe resume"):
        runner.evaluate(changed)
    assert len(fake_runtime["assistant"][2].calls) == 0

    changed.force = True
    forced = runner.evaluate(changed)
    assert forced.ok
    rewritten = json.loads((output / "transcripts" / "S1.json").read_text())
    assert rewritten["assistant_model"] == "model-b"
    assert len(fake_runtime["assistant"][3].calls) == 5


def test_corrupt_existing_transcript_fails_closed_but_force_replaces_it(
    tmp_path: Path,
    fake_runtime: dict[str, list[FakeClient]],
) -> None:
    output = tmp_path / "run"
    config = EvalConfig(
        samples=[make_sample()],
        model_under_test="model-a",
        user_model="user-a",
        judge="skip",
        output_dir=output,
        workers=1,
    )
    runner.evaluate(config)
    transcript = output / "transcripts" / "S1.json"
    transcript.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing unsafe resume"):
        runner.evaluate(config)
    config.force = True
    assert runner.evaluate(config).ok
    assert json.loads(transcript.read_text())["scenario_id"] == "S1"


@pytest.mark.parametrize(
    "samples,sample_ids",
    [
        ([make_sample(), make_sample()], None),
        ([make_sample()], ["S1", "S1"]),
    ],
)
def test_duplicate_selected_ids_are_rejected(
    tmp_path: Path,
    fake_runtime: dict[str, list[FakeClient]],
    samples: list[dict[str, Any]],
    sample_ids: list[str] | None,
) -> None:
    with pytest.raises(ValueError, match="duplicate selected sample ids"):
        runner.evaluate(
            EvalConfig(
                samples=samples,
                sample_ids=sample_ids,
                output_dir=tmp_path / "run",
                judge="skip",
            )
        )


def test_samples_loader_accepts_file_or_run_directory(tmp_path: Path) -> None:
    direct = tmp_path / "direct.jsonl"
    write_samples(direct, [make_sample()])
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    write_samples(run_dir / "samples.jsonl", [make_sample("S2")])
    assert runner._load_samples(direct)[0]["id"] == "S1"
    assert runner._load_samples(run_dir)[0]["id"] == "S2"


def test_cli_propagates_service_tier_seed_and_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[EvalConfig] = []

    def fake_evaluate(config: EvalConfig) -> EvalReport:
        captured.append(config)
        return EvalReport(output_dir=Path(config.output_dir), judge_status="skipped")

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    result = cli.main(
        [
            "--samples",
            str(tmp_path / "samples.jsonl"),
            "--service-tier",
            "priority",
            "--rollout-seed",
            "17",
            "--force",
        ]
    )
    assert result == 0
    assert captured[0].service_tier == "priority"
    assert captured[0].rollout_seed == 17
    assert captured[0].force is True


class FakeHTTPResponse:
    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ).encode()


@pytest.mark.parametrize("kind", ["openrouter", "local", "custom"])
def test_clients_forward_seed_and_local_clients_accept_cache_tail(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        payloads.append(json.loads(request.data))
        return FakeHTTPResponse()

    monkeypatch.setattr(pipeline_client.urllib.request, "urlopen", fake_urlopen)
    if kind == "openrouter":
        client: Any = pipeline_client.OpenRouterClient(
            api_key="key", model="provider/model", retries=1
        )
    elif kind == "local":
        client = pipeline_client.LocalOpenAIClient(
            model=pipeline_client.LOCAL_USER_MODEL, retries=1
        )
    else:
        client = CustomVLLMClient(
            base_url="http://example.test/v1",
            served_model_name="served-model",
            api_key="key",
            retries=1,
        )
        assert client.usage["calls"] == 0
        assert client.service_tier is None
        assert isinstance(client._usage_lock, type(threading.Lock()))

    assert client.complete(
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.1,
        max_tokens=10,
        cache_tail=False,
        seed=42,
    ) == "ok"
    assert payloads[0]["seed"] == 42


def test_partial_and_failed_judges_make_report_unsuccessful(
    tmp_path: Path,
    fake_runtime: dict[str, list[FakeClient]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import figsbench.judge as judge_package

    monkeypatch.setattr(
        judge_package,
        "run_judge",
        lambda **kwargs: {
            "status": "incomplete",
            "counts": {"success": 0, "api_error": 1},
        },
    )
    report = runner.evaluate(
        EvalConfig(
            samples=[make_sample()],
            model_under_test="model-a",
            user_model="user-a",
            judge="api",
            output_dir=tmp_path / "partial",
            workers=1,
        )
    )
    assert report.judge_status == "incomplete"
    assert not report.ok

    def fail_judge(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(judge_package, "run_judge", fail_judge)
    report = runner.evaluate(
        EvalConfig(
            samples=[make_sample("S2")],
            model_under_test="model-a",
            user_model="user-a",
            judge="api",
            output_dir=tmp_path / "failed",
            workers=1,
        )
    )
    assert report.judge_status == "failed"
    assert "judge unavailable" in str(report.judge_error)
    assert not report.ok


def test_complete_judge_is_published_and_metrics_are_computed(
    tmp_path: Path,
    fake_runtime: dict[str, list[FakeClient]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import figsbench.judge as judge_package


    def complete_judge(**kwargs: Any) -> dict[str, Any]:
        result = {
            "sample_id": "S1",
            "status": "success",
            "judgment": {
                "sycophancy": {"score": 1},
                "calibrated_validation": {"score": 2},
            },
        }
        result_dir = Path(kwargs["out_root"]) / kwargs["judge_run_id"] / "results"
        result_dir.mkdir(parents=True)
        (result_dir / "S1.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "complete", "counts": {"success": 1}}

    monkeypatch.setattr(judge_package, "run_judge", complete_judge)
    output = tmp_path / "complete"
    config = EvalConfig(
        samples=[make_sample()],
        model_under_test="model-a",
        user_model="user-a",
        judge="api",
        output_dir=output,
        workers=1,
    )
    report = runner.evaluate(config)
    assert report.ok
    assert report.judge_status == "complete"
    assert report.judge_run_id is not None
    assert "-judged-api-" in report.judge_run_id
    assert len(report.judge_run_id.rsplit("-", 1)[-1]) == 24
    assert (output / "metrics.json").is_file()
    assert json.loads((output / "metrics.json").read_text())["benchmark_score"] == 100.0
    assert (output / "per_sample_scores.jsonl").is_file()
    copied = (
        output
        / "judgments"
        / report.judge_run_id
        / "results"
        / "S1.json"
    )
    assert copied.is_file()
    assert json.loads((output / "judgments" / "latest.json").read_text())[
        "judge_run_id"
    ] == report.judge_run_id

    api_judge_run_id = report.judge_run_id
    config.judge = "local"
    local_report = runner.evaluate(config)
    assert local_report.ok
    assert local_report.judge_run_id is not None
    assert "-judged-local-" in local_report.judge_run_id
    assert local_report.judge_run_id != api_judge_run_id


def test_config_validation_and_serialization_do_not_persist_secrets() -> None:
    config = EvalConfig(
        samples=[], endpoint_api_key="secret", force=True, rollout_seed=10
    )
    serialized = json.loads(config.to_json())
    assert "endpoint_api_key" not in serialized
    assert "force" not in serialized
    assert serialized["rollout_seed"] == 10
    with pytest.raises(ValueError, match="rollout_seed"):
        EvalConfig(samples=[], rollout_seed=True).validate()


def _capture_complete_judge(seen: list[dict[str, Any]]) -> Any:
    def complete_judge(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        result = {
            "sample_id": "S1",
            "status": "success",
            "judgment": {
                "sycophancy": {"score": 1},
                "calibrated_validation": {"score": 1},
            },
        }
        result_dir = Path(kwargs["out_root"]) / kwargs["judge_run_id"] / "results"
        result_dir.mkdir(parents=True)
        (result_dir / "S1.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "complete", "counts": {"success": 1}}

    return complete_judge


@pytest.mark.parametrize(
    ("judge", "judge_model", "expected_model", "paper_judge"),
    [
        ("api", None, "google/gemini-3.8-flash", False),
        ("api", "z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash", False),
        ("local", None, "local-glm-5.3-flash", True),
    ],
)
def test_judge_model_selection_is_recorded(
    tmp_path: Path,
    fake_runtime: dict[str, list[FakeClient]],
    monkeypatch: pytest.MonkeyPatch,
    judge: str,
    judge_model: str | None,
    expected_model: str,
    paper_judge: bool,
) -> None:
    import figsbench.judge as judge_package

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        judge_package, "run_judge", _capture_complete_judge(seen)
    )
    output = tmp_path / "out"
    report = runner.evaluate(
        EvalConfig(
            samples=[make_sample()],
            model_under_test="model-a",
            user_model="user-a",
            judge=judge,
            judge_model=judge_model,
            output_dir=output,
            workers=1,
        )
    )
    assert report.ok
    assert [call["judge_model"] for call in seen] == [expected_model]
    metrics = json.loads((output / "metrics.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    for record in (metrics, manifest):
        assert record["judge_backend"] == judge
        assert record["judge_model"] == expected_model
        assert record["paper_judge"] is paper_judge
        assert record["user_model"] == "user-a"
        assert record["paper_user_simulator"] is False
        assert record["paper_protocol"] is False


@pytest.mark.parametrize("judge", ["local", "skip"])
def test_judge_model_requires_api_judge(judge: str) -> None:
    with pytest.raises(ValueError, match="judge_model"):
        EvalConfig(
            samples=[make_sample()], judge=judge, judge_model="google/gemini-3.8-flash"
        ).validate()


def test_cli_passes_judge_model() -> None:
    from figsbench.evaluation import __main__ as cli

    args = cli.parse_args(
        ["--samples", "x", "--judge", "api", "--judge-model", "z-ai/glm-5.3-flash"]
    )
    assert args.judge == "api"
    assert args.judge_model == "z-ai/glm-5.3-flash"


def test_user_model_api_resolves_to_paper_weights_on_openrouter() -> None:
    from figsbench.evaluation.config import API_USER_MODEL, PAPER_USER_MODEL

    assert EvalConfig(samples=[]).user_model == PAPER_USER_MODEL == "local-deepseek-v4-flash"
    assert EvalConfig(samples=[], user_model="api").user_model == API_USER_MODEL
    assert API_USER_MODEL == "deepseek/deepseek-v4-flash"


@pytest.mark.parametrize(
    ("model", "pinned"),
    [
        ("deepseek/deepseek-v4-flash", True),
        ("deepseek/deepseek-v4-flash:floor", True),
        ("deepseek/deepseek-v4-flash-0731", False),
        ("~deepseek/deepseek-v4-flash-latest", False),
        ("deepseek/deepseek-v4-flash-vision-exp", False),
    ],
)
def test_v4_flash_provider_pin_is_exact(model: str, pinned: bool) -> None:
    prefs = pipeline_client.openrouter_provider_preferences(model)
    assert (prefs == pipeline_client.DEEPSEEK_V4_FLASH_PROVIDER_PREFERENCES) is pinned
