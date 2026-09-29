"""Model clients for evaluation: OpenRouter, self-hosted aliases, custom vLLM."""

from __future__ import annotations

import threading

from benchmark.pipeline.client import (
    LocalOpenAIClient,
    OpenRouterClient,
    build_client,
    new_usage,
)


class CustomVLLMClient(LocalOpenAIClient):
    """OpenAI-compatible client for an arbitrary self-hosted endpoint."""

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
    """Build the client for the model under test.

    Custom endpoint wins when given; otherwise `local-*` aliases route to
    self-hosted servers and everything else goes to OpenRouter.
    """
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
    if reasoning_effort is not None and not isinstance(
        client, LocalOpenAIClient
    ):
        client.reasoning_effort = reasoning_effort
    return client
