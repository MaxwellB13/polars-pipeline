"""A ``Reader`` around any function that returns a frame.

For sources that are "call this and get a DataFrame back" -- an internal
database client, an API wrapper, a notebook helper -- rather than a file.

The awkward part of such sources is *staleness*: a file has an mtime, a query
result has nothing observable. So the fingerprint is built from ``key`` (what
the call *is*: SQL text, parameters, endpoint) and, optionally, ``watermark``
-- a cheap probe of what the call *would return* (``SELECT MAX(updated_at)``,
a row count, an ETag). With a watermark, ``refresh="stale"`` re-ingests only
when the upstream data moved. Without one, "stale" means "the query changed",
and you re-ingest an updated database with ``refresh={"name"}`` or ``"all"``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from polars_pipeline.errors import SourceError


@dataclass(frozen=True)
class FunctionReader:
    fn: Callable[[], pl.DataFrame | pl.LazyFrame]
    key: Any = None
    watermark: Callable[[], Any] | None = None
    name: str = field(default="")

    def describe(self) -> str:
        label = self.name or getattr(self.fn, "__qualname__", repr(self.fn))
        return f"call:{label}"

    def fingerprint(self) -> str:
        payload = {"key": _jsonable(self.key), "describe": self.describe()}
        if self.watermark is not None:
            try:
                payload["watermark"] = _jsonable(self.watermark())
            except Exception as exc:
                raise SourceError(f"{self.describe()}: watermark probe failed: {exc}") from exc
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def scan(self) -> pl.LazyFrame:
        out = self.fn()
        if isinstance(out, pl.DataFrame):
            return out.lazy()
        if isinstance(out, pl.LazyFrame):
            return out
        raise SourceError(f"{self.describe()}: expected a polars frame, got {type(out).__name__}")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return repr(obj)
