"""Transport unit fixtures only; no real provider or model experiment."""

import io
import json
import urllib.error

import pytest

from tools.pilot_transport import (
    MODEL,
    AuditedProvider,
    ProviderContentError,
    ProviderInfrastructureError,
)


class Response(io.BytesIO):
    status = 200


def outer(content):
    return json.dumps(
        {
            "model": MODEL,
            "id": "fixture-id",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    ).encode()


def test_raw_preserved_and_credential_never_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "PRIVATE_CREDENTIAL_UNIT_FIXTURE")
    raw = outer('{"status":"ok"}')
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(raw))
    provider = AuditedProvider(tmp_path)
    result = provider.complete(tokens=(), messages=[], response_format={})
    assert result.response == {"status": "ok"}
    assert (tmp_path / "attempt_0.response.raw").read_bytes() == raw
    assert all(b"PRIVATE_CREDENTIAL_UNIT_FIXTURE" not in p.read_bytes() for p in tmp_path.iterdir())
    with pytest.raises(RuntimeError):
        provider.complete(tokens=(), messages=[], response_format={})


def test_transport_retry_identical_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    requests = []

    def send(request, **kwargs):
        requests.append(request.data)
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "fixture", {}, io.BytesIO(b"busy"))
        return Response(outer("{}"))

    monkeypatch.setattr("urllib.request.urlopen", send)
    monkeypatch.setattr("time.sleep", lambda _: None)
    AuditedProvider(tmp_path).complete(tokens=(), messages=[], response_format={})
    assert len(requests) == 2 and requests[0] == requests[1]
    metadata = json.loads((tmp_path / "transport.json").read_text())
    assert metadata["retry_count"] == 1 and metadata["provider_error_count"] == 1


def test_malformed_content_no_retry_or_repair(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    count = []

    def send(*args, **kwargs):
        count.append(1)
        return Response(outer("```json\n{}\n```"))

    monkeypatch.setattr("urllib.request.urlopen", send)
    with pytest.raises(ProviderContentError):
        AuditedProvider(tmp_path).complete(tokens=(), messages=[], response_format={})
    assert len(count) == 1
    assert (tmp_path / "message.json").exists() and (tmp_path / "attempt_0.response.raw").exists()


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        b'{"model":"wrong","id":"fixture","choices":[{"message":{"content":"{}"}}]}',
        b'{"model":"google/gemini-2.5-flash","id":"fixture","choices":[]}',
        b'{"model":"google/gemini-2.5-flash","id":"fixture","choices":[{"message":{"content":null}}]}',
    ],
)
def test_malformed_outer_is_infrastructure_not_correctable_content(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    calls = []

    def send(*args, **kwargs):
        calls.append(1)
        return Response(raw)

    monkeypatch.setattr("urllib.request.urlopen", send)
    with pytest.raises(ProviderInfrastructureError):
        AuditedProvider(tmp_path).complete(tokens=(), messages=[], response_format={})
    assert len(calls) == 1
    assert (tmp_path / "attempt_0.response.raw").read_bytes() == raw
    assert (tmp_path / "protocol_error.json").is_file()
    assert not (tmp_path / "schema_error.json").exists()
    transport = json.loads((tmp_path / "transport.json").read_text())
    assert transport["protocol_error_count"] == 1
    assert transport["provider_error_count"] == 0


@pytest.mark.parametrize("error_location", ["top_level", "choice", "finish_reason"])
def test_http_200_provider_error_fails_closed(tmp_path, monkeypatch, error_location):
    """Synthetic infrastructure fixtures, not benchmark/model-generation tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    payload = json.loads(outer('{"status":"ok"}'))
    error = {"code": 400, "message": "fixture non-retryable error"}
    if error_location == "top_level":
        payload["error"] = error
    elif error_location == "choice":
        payload["choices"][0]["error"] = error
    else:
        payload["choices"][0]["finish_reason"] = "error"
    raw = json.dumps(payload).encode()
    calls = []

    def send(*args, **kwargs):
        calls.append(1)
        return Response(raw)

    monkeypatch.setattr("urllib.request.urlopen", send)
    with pytest.raises(ProviderInfrastructureError):
        AuditedProvider(tmp_path).complete(tokens=(), messages=[], response_format={})
    assert len(calls) == 1
    assert (tmp_path / "attempt_0.response.raw").read_bytes() == raw
    assert (tmp_path / "response.json").exists()
    assert (tmp_path / "provider_envelope_error.json").exists()
    assert not (tmp_path / "parsed.json").exists()
    assert not (tmp_path / "schema_error.json").exists()
    transport = json.loads((tmp_path / "transport.json").read_text())
    assert transport["provider_error_count"] == 1
    assert transport["retry_count"] == 0


@pytest.mark.parametrize("second", ["success", "rate_limit", "timeout"])
def test_envelope_rate_limit_retries_identical_body_once(tmp_path, monkeypatch, second):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    payload = json.loads(outer("not model JSON"))
    payload["choices"][0].update(
        error={"code": 429, "message": "fixture rate limit"}, finish_reason="error"
    )
    rate_limit = json.dumps(payload).encode()
    requests = []
    sleeps = []

    def send(request, **kwargs):
        requests.append(request.data)
        if len(requests) == 1 or second == "rate_limit":
            return Response(rate_limit)
        if second == "timeout":
            raise TimeoutError("fixture timeout")
        return Response(outer('{"status":"ok"}'))

    monkeypatch.setattr("urllib.request.urlopen", send)
    monkeypatch.setattr("time.sleep", sleeps.append)
    provider = AuditedProvider(tmp_path)
    if second == "success":
        result = provider.complete(tokens=(), messages=[], response_format={})
        assert result.response == {"status": "ok"}
    else:
        with pytest.raises(ProviderInfrastructureError):
            provider.complete(tokens=(), messages=[], response_format={})
        assert not (tmp_path / "parsed.json").exists()
    assert len(requests) == 2 and requests[0] == requests[1]
    assert sleeps == [5]
    assert (tmp_path / "attempt_0.response.raw").read_bytes() == rate_limit
    assert (tmp_path / "attempt_0.provider_error.json").exists()
    assert not (tmp_path / "schema_error.json").exists()
    metadata = json.loads((tmp_path / "transport.json").read_text())
    assert metadata["retry_count"] == 1
    assert metadata["provider_error_count"] == (1 if second == "success" else 2)
    assert metadata["request_sha256"]
