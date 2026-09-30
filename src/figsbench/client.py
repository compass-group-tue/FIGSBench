"""Small dependency-free OpenRouter client used by generation stages."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


class _UnsetType:
    """Distinguish an omitted per-call option from an explicit ``None``."""

    __slots__ = ()


_UNSET = _UnsetType()

# OpenRouter provider slug for fast inference (see openrouter.ai/provider/coreweave).
COREWEAVE_FP4_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["coreweave"],
    "quantizations": ["fp4"],
    "allow_fallbacks": False,
}
KIMI_K26_PROVIDER_PREFERENCES = COREWEAVE_FP4_PROVIDER_PREFERENCES
GLM_52_PROVIDER_PREFERENCES = COREWEAVE_FP4_PROVIDER_PREFERENCES
GLM_53_FLASH_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["z-ai", "novita", "nextbit", "reka", "gmicloud"],
    "ignore": ["baseten", "modal", "fireworks"],
    "allow_fallbacks": True,
}
GLM_53_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["together"],
    "allow_fallbacks": False,
}
INKLING_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["baseten"],
    "quantizations": ["fp8"],
    "allow_fallbacks": False,
}
FABLE_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["anthropic"],
    "allow_fallbacks": False,
}
ASTRA_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["openai"],
    "allow_fallbacks": False,
}
GEMINI_38_FLASH_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["google-ai-studio"],
    "allow_fallbacks": False,
}
DEEPSEEK_V4_FLASH_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["alibaba"],
    "quantizations": ["fp8"],
    "allow_fallbacks": False,
}
DEEPSEEK_V41_FLASH_PROVIDER_PREFERENCES: dict[str, Any] = {
    "order": ["siliconflow"],
    "quantizations": ["fp8"],
    "allow_fallbacks": False,
}


def openrouter_provider_preferences(model: str) -> dict[str, Any] | None:
    """Return OpenRouter provider routing for models with a known fast route."""

    if os.environ.get("OPENROUTER_DISABLE_PROVIDER_ROUTING", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return None
    override_order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
    if override_order:
        prefs: dict[str, Any] = {
            "order": [s.strip() for s in override_order.split(",") if s.strip()],
            "allow_fallbacks": os.environ.get(
                "OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false").lower() in (
                "1", "true", "yes"),
        }
        override_quant = os.environ.get(
            "OPENROUTER_PROVIDER_QUANTIZATIONS", "").strip()
        if override_quant:
            prefs["quantizations"] = [
                s.strip() for s in override_quant.split(",") if s.strip()]
        return prefs
    normalized = model.strip().lower()
    if "claude-fable-5.1" in normalized:
        return dict(FABLE_PROVIDER_PREFERENCES)
    if "gpt-6-astra" in normalized:
        return dict(ASTRA_PROVIDER_PREFERENCES)
    if "kimi-k2.6" in normalized:
        return dict(KIMI_K26_PROVIDER_PREFERENCES)
    if "glm-5.2" in normalized:
        return dict(GLM_52_PROVIDER_PREFERENCES)
    if "glm-5.3-flash" in normalized:
        return dict(GLM_53_FLASH_PROVIDER_PREFERENCES)
    if "glm-5.3" in normalized:
        return dict(GLM_53_PROVIDER_PREFERENCES)
    if "deepseek-v4.1-flash" in normalized or "deepseek-v4-1-flash" in normalized:
        return dict(DEEPSEEK_V41_FLASH_PROVIDER_PREFERENCES)
    # Exact id only: the Alibaba fp8 route serves DeepSeek-V4-Flash 0423 (the paper's weights,
    # deepseek-ai/DeepSeek-V4-Flash), not -0731, -latest or -vision-exp.
    if normalized.split(":", 1)[0].rsplit("/", 1)[-1] == "deepseek-v4-flash":
        return dict(DEEPSEEK_V4_FLASH_PROVIDER_PREFERENCES)
    if "inkling" in normalized:
        return dict(INKLING_PROVIDER_PREFERENCES)
    if "gemini-3.8-flash" in normalized or "gemini-3-8-flash" in normalized:
        return dict(GEMINI_38_FLASH_PROVIDER_PREFERENCES)
    return None


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def new_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
            "reasoning_tokens": 0, "calls": 0}


def _accumulate(client: Any, usage: dict[str, Any]) -> None:
    try:
        details = usage.get("prompt_tokens_details") or {}
        cdetails = usage.get("completion_tokens_details") or {}
        inc = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cached_tokens": int(details.get("cached_tokens", 0) or 0),
            "cache_write_tokens": int(details.get("cache_write_tokens", 0) or 0),
            "reasoning_tokens": int(cdetails.get("reasoning_tokens", 0) or 0),
            "calls": 1,
        }
    except (TypeError, ValueError):
        return
    with client._usage_lock:
        for k, v in inc.items():
            client.usage[k] = client.usage.get(k, 0) + v


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        retries: int = 5,
        reasoning_effort: str | None = None,
        provider: dict[str, Any] | None = None,
        service_tier: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An OpenRouter API key is required")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.reasoning_effort = reasoning_effort
        self.service_tier = (
            service_tier
            or os.environ.get("OPENROUTER_SERVICE_TIER", "").strip()
            or None
        )
        if provider is not None:
            self.provider = provider
        else:
            self.provider = openrouter_provider_preferences(model)
        self.usage = new_usage()
        self._usage_lock = threading.Lock()

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]] | None = None,
        temperature: float,
        max_tokens: int | None,
        json_output: bool = False,
        reasoning_effort: str | None | _UnsetType = _UNSET,
        timeout: float | None | _UnsetType = _UNSET,
        cache_tail: bool = True,
        seed: int | None = None,
    ) -> str:
        selected_reasoning = (
            self.reasoning_effort
            if reasoning_effort is _UNSET
            else reasoning_effort
        )
        selected_timeout = self.timeout if timeout is _UNSET else timeout
        normalized_model = self.model.strip().lower()
        if selected_reasoning is None and (
            "gemini-3.8-flash" in normalized_model or "glm-5.3" in normalized_model
        ):
            selected_reasoning = "minimal"
        json_mode_supported = "inkling" not in normalized_model
        if "claude" in normalized_model and isinstance(system, str):
            system_content: Any = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_content = system
        ordered_messages = [{"role": "system", "content": system_content}, *(messages or [])]
        if "claude" in normalized_model and cache_tail and ordered_messages:
            last = ordered_messages[-1]
            if isinstance(last.get("content"), str):
                last["content"] = [
                    {
                        "type": "text",
                        "text": last["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ordered_messages,
            "temperature": temperature,
            **({} if max_tokens is None else {"max_tokens": max_tokens}),
            "reasoning": (
                {"enabled": False, "exclude": True}
                if selected_reasoning is None
                else {"effort": selected_reasoning, "exclude": True}
            ),
        }
        if seed is not None:
            payload["seed"] = seed
        if json_output and json_mode_supported:
            payload["response_format"] = {"type": "json_object"}
        if self.provider:
            payload["provider"] = self.provider
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/compass-group-tue/FIGSBench",
                "X-Title": "FIGSBench",
            },
        )
        last_error: Exception | None = None
        selected_max_tokens = max_tokens
        for attempt in range(self.retries):
            try:
                if selected_max_tokens is not None and selected_max_tokens != payload.get("max_tokens"):
                    payload["max_tokens"] = selected_max_tokens
                    request = urllib.request.Request(
                        OPENROUTER_URL,
                        data=json.dumps(payload).encode("utf-8"),
                        method="POST",
                        headers=dict(request.header_items()),
                    )
                with urllib.request.urlopen(request, timeout=selected_timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("OpenRouter returned empty content")
                _accumulate(self, result.get("usage") or {})
                return content.strip()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                ValueError,
            ) as exc:
                last_error = exc
                if isinstance(exc, ValueError) and "empty content" in str(exc):
                    if selected_max_tokens is None:
                        pass
                    else:
                        selected_max_tokens = min(
                            selected_max_tokens * 2, EMPTY_BUDGET_ESCALATION_CAP
                        )
                _retry_sleep(attempt, self.retries, self.model, exc)
        raise RuntimeError(
            f"OpenRouter request failed after {self.retries} attempts: {last_error}"
        ) from last_error

    def complete_json(
        self,
        *,
        system: str,
        messages: list[dict[str, str]] | None = None,
        temperature: float,
        max_tokens: int,
        attempts: int = 3,
        validator: Callable[[dict[str, Any]], None] | None = None,
        reasoning_effort: str | None | _UnsetType = _UNSET,
        timeout: float | None | _UnsetType = _UNSET,
    ) -> dict[str, Any]:
        """Request and parse JSON, retrying model-level formatting failures."""
        last_error: Exception | None = None
        corrective = ""
        for _ in range(attempts):
            text = self.complete(
                system=system + corrective,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_output=True,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
            try:
                value = parse_json_object(text)
                if validator:
                    validator(value)
                return value
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                detail = " ".join(str(exc).split())[:500]
                corrective = (
                    "\n\nIMPORTANT: Your previous response did not satisfy the required "
                    f"JSON contract: {detail}. Correct that exact defect and return one "
                    "complete JSON object only, with ordinary ASCII JSON braces."
                )
        raise RuntimeError(
            f"Model did not return valid JSON after {attempts} attempts: {last_error}"
        ) from last_error


def _normalize_model_json(text: str) -> str:
    """Repair common GLM-5.2 JSON formatting defects before parsing."""
    import re

    stripped = text.strip()
    # Leading brace duplicated: {"{...} or {\n{"...
    if stripped.startswith('{"{"'):
        stripped = "{" + stripped[3:]
    elif stripped.startswith("{\n{\"") or stripped.startswith("{\r\n{\""):
        stripped = "{" + stripped[stripped.index('{"') + 1 :]
    # Repeated key without colon: {"score{"score": ...}
    m = re.match(r'^\{"(\w+)\{"(\w+)":', stripped)
    if m and m.group(1) == m.group(2):
        stripped = '{"' + m.group(1) + '":' + stripped[m.end() :]
    # Nested duplicate envelope: {"score":{"score":2,...}}
    if stripped.startswith('{"score":{') and '"score"' in stripped[10:30]:
        inner_start = stripped.find('{"score":', 1)
        if inner_start > 0:
            stripped = stripped[inner_start:]
    return stripped


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model response, tolerating Markdown fences or surrounding prose."""
    stripped = _normalize_model_json(text)
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = stripped.find("{")
        if start < 0:
            raise
        value, _ = decoder.raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from model")
    return value


LOCAL_GLM_MODEL = "local-glm-5.3-flash"
LOCAL_USER_MODEL = "local-deepseek-v4-flash"

LOCAL_MODEL_ROUTES: dict[str, tuple[str, str]] = {
    LOCAL_GLM_MODEL: (
        "LOCAL_GLM_BASE_URL",
        "http://localhost:8000/v1",
    ),
    LOCAL_USER_MODEL: (
        "LOCAL_USER_BASE_URL",
        "http://localhost:8001/v1",
    ),
}

LOCAL_MODEL_SERVED_NAMES: dict[str, tuple[str, str]] = {
    LOCAL_GLM_MODEL: ("LOCAL_GLM_MODEL", "zai-org/GLM-5.3-Flash"),
    LOCAL_USER_MODEL: ("LOCAL_USER_MODEL", "deepseek-v4-flash"),
}


def is_local_model(model: str) -> bool:
    return model.strip().lower() in LOCAL_MODEL_ROUTES


EMPTY_BUDGET_ESCALATION_CAP = 64000


def _retry_sleep(attempt: int, retries: int, model: str, error: Exception) -> None:
    import sys

    if attempt + 1 >= retries:
        return
    delay = min(20.0, (2**attempt) + random.random())
    print(
        f"RETRY {model} attempt {attempt + 1}/{retries} after "
        f"{type(error).__name__}: {str(error)[:200]} (sleep {delay:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(delay)


class LocalOpenAIClient(OpenRouterClient):
    """OpenAI-compatible client for self-hosted vLLM servers (no API key)."""

    def __init__(
        self,
        model: str,
        timeout: float = 180.0,
        retries: int = 5,
        base_url: str | None = None,
        served_model_name: str | None = None,
    ) -> None:
        key = model.strip().lower()
        url_env, url_default = LOCAL_MODEL_ROUTES[key]
        name_env, name_default = LOCAL_MODEL_SERVED_NAMES[key]
        self.base_url = (base_url or os.environ.get(url_env) or url_default).rstrip("/")
        self.served_model_name = served_model_name or os.environ.get(name_env) or name_default
        self.api_key = os.environ.get("LOCAL_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.reasoning_effort = None
        self.provider = None
        self.usage = new_usage()
        self._usage_lock = threading.Lock()

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]] | None = None,
        temperature: float,
        max_tokens: int | None,
        json_output: bool = False,
        reasoning_effort: str | None | _UnsetType = _UNSET,
        timeout: float | None | _UnsetType = _UNSET,
        cache_tail: bool = True,
        seed: int | None = None,
    ) -> str:
        # ``cache_tail`` is an OpenRouter optimization.  Accept it here so
        # callers can use one completion interface for hosted and local
        # OpenAI-compatible endpoints; local servers simply ignore it.
        del cache_tail
        selected_timeout = self.timeout if timeout is _UNSET else timeout
        payload: dict[str, Any] = {
            "model": self.served_model_name,
            "messages": [{"role": "system", "content": system}, *(messages or [])],
            "temperature": temperature,
            **({} if max_tokens is None else {"max_tokens": max_tokens}),
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        last_error: Exception | None = None
        selected_max_tokens = max_tokens
        for attempt in range(self.retries):
            try:
                if selected_max_tokens is not None and selected_max_tokens != payload.get("max_tokens"):
                    payload["max_tokens"] = selected_max_tokens
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    request = urllib.request.Request(
                        self.base_url + "/chat/completions",
                        data=json.dumps(payload).encode("utf-8"),
                        method="POST",
                        headers=headers,
                    )
                with urllib.request.urlopen(request, timeout=selected_timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Local server returned empty content")
                _accumulate(self, result.get("usage") or {})
                return content.strip()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                ValueError,
            ) as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        body = exc.read().decode("utf-8", "replace")[:500]
                    except Exception:
                        body = ""
                    exc._body_note = body
                last_error = exc
                if isinstance(exc, ValueError) and "empty content" in str(exc):
                    if selected_max_tokens is None:
                        pass
                    else:
                        selected_max_tokens = min(
                            selected_max_tokens * 2, EMPTY_BUDGET_ESCALATION_CAP
                        )
                _retry_sleep(attempt, self.retries, self.model, exc)
        note = getattr(last_error, "_body_note", "")
        raise RuntimeError(
            f"Local request to {self.base_url} failed after {self.retries} attempts: {last_error}"
            + (f" body={note}" if note else "")
        ) from last_error


def build_client(
    model: str,
    api_key: str = "",
    timeout: float = 180.0,
    retries: int = 5,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
) -> OpenRouterClient:
    """Route `local-*` models to self-hosted servers, everything else to OpenRouter."""
    if is_local_model(model):
        return LocalOpenAIClient(model=model, timeout=timeout, retries=retries)
    return OpenRouterClient(
        api_key=api_key,
        model=model,
        timeout=timeout,
        retries=retries,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
    )


class CustomVLLMClient(LocalOpenAIClient):
    """OpenAI-compatible client for an arbitrary self-hosted endpoint (model under test)."""

    def __init__(
        self,
        *,
        base_url: str,
        served_model_name: str,
        api_key: str = "",
        timeout: float = 180.0,
        retries: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_model_name = served_model_name
        self.api_key = api_key
        self.model = served_model_name
        self.timeout = timeout
        self.retries = retries
        self.reasoning_effort = None
        self.provider = None
        self.service_tier = None
        self.usage = new_usage()
        self._usage_lock = threading.Lock()


def build_eval_client(
    *,
    model: str,
    api_key: str = "",
    endpoint_url: str | None = None,
    served_model: str | None = None,
    endpoint_api_key: str = "",
    reasoning_effort: str | None = None,
    timeout: float = 180.0,
    retries: int = 5,
    service_tier: str | None = None,
) -> OpenRouterClient:
    """Client for the model under test: a custom endpoint, a `local-*` alias, or OpenRouter."""
    if endpoint_url:
        return CustomVLLMClient(
            base_url=endpoint_url,
            served_model_name=served_model or model,
            api_key=endpoint_api_key,
            timeout=timeout,
            retries=retries,
        )
    if service_tier is None and "gpt-6-astra" in (model or "").strip().lower():
        service_tier = "flex"
    client = build_client(
        model,
        api_key=api_key,
        timeout=timeout,
        retries=retries,
        service_tier=service_tier,
    )
    if reasoning_effort is not None and not isinstance(client, LocalOpenAIClient):
        client.reasoning_effort = reasoning_effort
    return client
