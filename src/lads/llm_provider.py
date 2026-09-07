"""Shared LLM transport used by condition-aware candidate generation."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class LLMResponseError(RuntimeError):
    """Fail-closed provider or response-boundary error."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LLMResponseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_content(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[len("```json") : -len("```")].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[len("```") : -len("```")].strip()
    return json.loads(stripped, object_pairs_hook=_reject_duplicate_pairs)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class LLMBatchCompletionV1:
    response: Mapping[str, Any]
    provider_name: str
    requested_model: str
    returned_model: str
    provider_response_id: str
    prompt_tokens: int
    completion_tokens: int
    request_payload_sha256: str
    raw_response_sha256: str
    response_mode: str = "json_schema"


class LLMBatchProviderV1(Protocol):
    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMBatchCompletionV1: ...


class InMemoryLLMBatchProviderV1:
    """Deterministic in-memory provider for boundary tests."""

    def __init__(
        self, response: Mapping[str, Any], *, model: str = "in-memory-llm-test-v1"
    ) -> None:
        self.response = response
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMBatchCompletionV1:
        request = {
            "tokens": list(tokens),
            "messages": list(messages),
            "response_format": response_format,
        }
        self.calls.append(request)
        return LLMBatchCompletionV1(
            self.response,
            "in-memory-test",
            self.model,
            self.model,
            "in-memory-response-1",
            1,
            1,
            _sha256(request),
            _sha256(self.response),
        )


class OpenRouterLLMBatchProviderV1:
    """Bounded stdlib-only OpenRouter structured-output transport."""

    def __init__(
        self,
        credential: Callable[[], str],
        *,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        use_json_schema: bool = True,
        max_output_tokens: int = 24_000,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        if not callable(credential):
            raise ValueError("credential must be callable")
        if not model or len(model) > 128:
            raise ValueError("model is invalid")
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError("timeout_seconds is invalid")
        if max_retries not in {0, 1, 2}:
            raise ValueError("max_retries is invalid")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 512 <= max_output_tokens <= 24_000
        ):
            raise ValueError("max_output_tokens is invalid")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= float(temperature) <= 2.0
        ):
            raise ValueError("temperature is invalid")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0.0 < float(top_p) <= 1.0
        ):
            raise ValueError("top_p is invalid")
        self._credential = credential
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.use_json_schema = bool(use_json_schema)
        self.max_output_tokens = max_output_tokens
        self.temperature = None if temperature is None else float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.request_count = 0

    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMBatchCompletionV1:
        if self.request_count:
            raise LLMResponseError("provider instance allows exactly one request")
        body_mapping: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "provider": {
                "require_parameters": self.use_json_schema,
                "allow_fallbacks": False,
            },
        }
        if self.use_json_schema:
            body_mapping["response_format"] = response_format
        if self.temperature is not None:
            body_mapping["temperature"] = self.temperature
        if self.top_p is not None:
            body_mapping["top_p"] = self.top_p
        completion_limit_key = (
            "max_tokens" if self.model.startswith("google/") else "max_completion_tokens"
        )
        body_mapping[completion_limit_key] = self.max_output_tokens
        body = _canonical_bytes(body_mapping)
        if len(body) > 1_048_576:
            raise LLMResponseError("provider request is oversized")
        key = self._credential()
        if not isinstance(key, str) or not key:
            raise LLMResponseError("provider credential is unavailable")
        request = urllib.request.Request(
            OPENROUTER_ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        self.request_count += 1
        raw: bytes | None = None
        last_error: Exception | None = None
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    with urllib.request.urlopen(  # nosec B310
                        request, timeout=self.timeout_seconds
                    ) as response:
                        raw = response.read(1_048_577)
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read(8_192).decode("utf-8", errors="replace")
                    last_error = LLMResponseError(f"provider HTTP {exc.code}: {detail[:1000]}")
                    if exc.code < 500 and exc.code != 429:
                        break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        finally:
            key = ""
        if raw is None:
            raise LLMResponseError(f"provider request failed: {last_error}") from last_error
        if len(raw) > 1_048_576:
            raise LLMResponseError("provider response is oversized")
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            outer = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
            choice = outer["choices"][0]
            content = choice["message"]["content"]
            returned_model = outer["model"]
            response_id = outer["id"]
            usage = outer.get("usage") or {}
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMResponseError("provider outer response is malformed") from exc
        if returned_model != self.model:
            raise LLMResponseError(
                f"provider model mismatch: requested {self.model}, got {returned_model}"
            )
        if not isinstance(content, str) or not isinstance(response_id, str):
            raise LLMResponseError("provider content or identity is invalid")
        try:
            parsed = _parse_json_content(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMResponseError("provider structured body is malformed") from exc
        return LLMBatchCompletionV1(
            parsed,
            "openrouter",
            self.model,
            returned_model,
            response_id,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            hashlib.sha256(body).hexdigest(),
            raw_sha256,
            "json_schema" if self.use_json_schema else "prompt_json_host_validated",
        )


__all__ = [
    "InMemoryLLMBatchProviderV1",
    "LLMBatchCompletionV1",
    "LLMBatchProviderV1",
    "LLMResponseError",
    "OpenRouterLLMBatchProviderV1",
]
