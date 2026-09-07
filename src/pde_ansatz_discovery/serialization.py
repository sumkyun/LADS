"""Strict project-owned canonical JSON and hashing utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


class CanonicalDataError(ValueError):
    """Raised when a value cannot cross the canonical data boundary."""


def freeze_json(value: Any, *, _active: set[int] | None = None) -> Any:
    """Deep-copy canonical JSON data into immutable containers.

    Cycles, non-string mapping keys, non-finite floats, sets, bytes, and arbitrary objects
    are rejected rather than stringified implicitly.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalDataError("non-finite floats are not canonical")
        return value

    active = _active if _active is not None else set()
    identity = id(value)
    if identity in active:
        raise CanonicalDataError("cyclic canonical data is not supported")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            copied: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise CanonicalDataError("canonical mapping keys must be strings")
                copied[key] = freeze_json(item, _active=active)
            return MappingProxyType(dict(sorted(copied.items())))
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return tuple(freeze_json(item, _active=active) for item in value)
    finally:
        active.remove(identity)
    raise CanonicalDataError(f"unsupported canonical value: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return a mutable JSON-compatible copy of frozen canonical data."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    raise CanonicalDataError(f"value is not frozen canonical data: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical data as sorted compact UTF-8 JSON with one trailing newline."""

    frozen = freeze_json(value)
    return (
        json.dumps(
            thaw_json(frozen),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CanonicalDataError",
    "canonical_json_bytes",
    "canonical_sha256",
    "freeze_json",
    "thaw_json",
]
