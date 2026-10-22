"""Audited hosted-LLM transport. No benchmark/reference or scoring imports."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from combined_solution_v1.llm_provider import (  # type: ignore[import-untyped]
    LLMBatchCompletionV1,
    LLMResponseError,
)

MODEL = "google/gemini-2.5-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRY_POLICY = "max1 identical-body transport retry after 5s; no provider/model fallback"


class ProviderInfrastructureError(BaseException):
    """Abort staged runner instead of swallowing infrastructure failures as candidates."""


class ProviderContentError(LLMResponseError):
    """The model message arrived, but its content is not a JSON object."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o444)


def write_json(path, value):
    write_bytes(path, canonical(value) + b"\n")


def credential():
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not value:
        raise ProviderInfrastructureError("OPENROUTER_API_KEY missing")
    return value


def duplicate_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key")
        result[k] = v
    return result


def provider_envelope_error(response):
    """HTTP success does not imply a successful upstream inference."""
    if not isinstance(response, Mapping):
        return None
    if response.get("error") is not None:
        return {"scope": "response", "error": response["error"]}
    choices = response.get("choices")
    if isinstance(choices, list):
        for index, choice in enumerate(choices):
            if not isinstance(choice, Mapping):
                continue
            if choice.get("error") is not None or choice.get("finish_reason") == "error":
                return {
                    "scope": "choice",
                    "choice_index": index,
                    "error": choice.get("error"),
                    "finish_reason": choice.get("finish_reason"),
                }
    return None


class AuditedProvider:
    def __init__(self, output, *, max_output_tokens=8192, model=MODEL, temperature=0, top_p=1):
        self.output = Path(output)
        self.model = model
        self.max_output_tokens = max_output_tokens
        # Defaults keep every existing caller on the frozen deterministic sampling policy.
        self.temperature = temperature
        self.top_p = top_p
        self.called = False

    def complete(self, *, tokens, messages, response_format):
        del tokens
        if self.called:
            raise RuntimeError("exactly one logical request per provider instance")
        self.called = True
        body = canonical(
            {
                "model": self.model,
                "messages": list(messages),
                "response_format": response_format,
                "provider": {"require_parameters": True, "allow_fallbacks": False},
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_output_tokens,
            }
        )
        write_bytes(self.output / "request.json", body)
        request = urllib.request.Request(
            ENDPOINT,
            data=body,
            method="POST",
            headers={"Authorization": "Bearer " + credential(), "Content-Type": "application/json"},
        )
        attempts = []
        raw = None
        envelope_error = None
        envelope = None
        for attempt in range(2):
            # Never reuse a previous failed response if the retry itself times out.
            raw = None
            envelope_error = None
            envelope = None
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    # Preserve the complete provider response before parsing or validation.
                    raw = response.read()
                    http_status = response.status
                write_bytes(self.output / f"attempt_{attempt}.response.raw", raw)
                try:
                    envelope = json.loads(raw, object_pairs_hook=duplicate_pairs)
                except (ValueError, UnicodeError):
                    # Preserve existing schema-error handling for malformed envelopes.
                    envelope = None
                envelope_error = provider_envelope_error(envelope)
                attempts.append(
                    {
                        "attempt": attempt,
                        "http_status": http_status,
                        "seconds": time.monotonic() - started,
                        "provider_error": envelope_error is not None,
                    }
                )
                if envelope_error is None:
                    break
                write_json(self.output / f"attempt_{attempt}.provider_error.json", envelope_error)
                error = envelope_error.get("error")
                code = str(error.get("code", "")) if isinstance(error, Mapping) else ""
                if code != "429" and not (code.isdigit() and 500 <= int(code) <= 599):
                    break
            except urllib.error.HTTPError as e:
                write_bytes(self.output / f"attempt_{attempt}.error.raw", e.read())
                attempts.append(
                    {
                        "attempt": attempt,
                        "http_status": e.code,
                        "seconds": time.monotonic() - started,
                        "provider_error": True,
                    }
                )
                if e.code < 500 and e.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                attempts.append(
                    {
                        "attempt": attempt,
                        "error_type": type(e).__name__,
                        "seconds": time.monotonic() - started,
                        "provider_error": True,
                    }
                )
            if attempt == 0:
                time.sleep(5)
        protocol_error = None
        outer = None
        if raw is not None and envelope_error is None:
            try:
                outer = json.loads(raw, object_pairs_hook=duplicate_pairs)
                if not isinstance(outer, Mapping):
                    raise ValueError("provider envelope must be a JSON object")
                write_json(self.output / "response.json", outer)
                returned = outer["model"]
                if returned != self.model:
                    raise ValueError("returned model mismatch; fallback forbidden")
                content = outer["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("non-string content")
                usage = outer.get("usage") or {}
                if not isinstance(usage, Mapping):
                    raise ValueError("provider usage must be an object")
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                response_id = outer["id"]
                if not isinstance(response_id, str):
                    raise ValueError("provider response id must be a string")
            except (KeyError, IndexError, TypeError, ValueError, UnicodeError) as error:
                protocol_error = {"type": type(error).__name__, "error": str(error)}
        write_json(
            self.output / "transport.json",
            {
                "logical_request_count": 1,
                "retry_count": len(attempts) - 1,
                "provider_error_count": sum(a["provider_error"] for a in attempts),
                "protocol_error_count": int(protocol_error is not None),
                "requested_model": self.model,
                "attempts": attempts,
                "request_sha256": hashlib.sha256(body).hexdigest(),
            },
        )
        if raw is None:
            raise ProviderInfrastructureError(
                "provider/transport failure; see saved transport metadata"
            )
        if envelope_error is not None:
            write_json(self.output / "response.json", envelope)
            write_json(self.output / "provider_envelope_error.json", envelope_error)
            raise ProviderInfrastructureError(
                "upstream error in HTTP-success envelope; raw response preserved; stop run"
            )
        if protocol_error is not None:
            write_json(self.output / "protocol_error.json", protocol_error)
            raise ProviderInfrastructureError(
                "malformed provider envelope; raw artifact preserved; stop run"
            )

        write_json(self.output / "message.json", {"content": content})
        try:
            # No fence stripping or host repair: the model's malformed content
            # remains a failed output, though a separately named method may ask
            # the provider for a fresh format-only correction request.
            parsed = json.loads(content, object_pairs_hook=duplicate_pairs)
            if not isinstance(parsed, Mapping):
                raise ValueError("model content must be a JSON object")
        except (TypeError, ValueError, UnicodeError) as e:
            write_json(self.output / "schema_error.json", {"type": type(e).__name__})
            raise ProviderContentError(
                "malformed model JSON content; raw artifact preserved"
            ) from e
        write_json(self.output / "parsed.json", parsed)
        return LLMBatchCompletionV1(
            parsed,
            "openrouter",
            self.model,
            returned,
            response_id,
            prompt_tokens,
            completion_tokens,
            hashlib.sha256(body).hexdigest(),
            hashlib.sha256(raw).hexdigest(),
            "json_schema",
        )
