"""Sources: where a pipeline's inputs come from, and how each becomes stage-ready.

Ingress for one source is always ``read -> prepare -> conform``:

1. ``reader.scan()`` gives the raw frame exactly as the file/DB has it.
2. ``prepare`` (any ``LazyFrame -> LazyFrame`` callables) cleans it up.
3. ``conform`` selects the spec's columns in spec order and casts them to the
   spec's dtypes, so what gets staged has the exact schema every downstream
   step and every hermetic run agrees on.

Anything that reads from somewhere new (a database, SharePoint, HTTP) only has
to implement the ``Reader`` protocol.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import polars as pl
from polspec import validate as polspec_validate
from polspec.tablespec import TableSpec, as_table_spec

from polars_pipeline.errors import SourceError
from polars_pipeline.sources.file import FileReader
from polars_pipeline.sources.function import FunctionReader
from polars_pipeline.sources.prep import Prepare, as_prepare_list

Conform = Literal["strict", "lenient", "none"]


@runtime_checkable
class Reader(Protocol):
    """Something that can produce a raw LazyFrame and say when it has changed."""

    def scan(self) -> pl.LazyFrame: ...

    def fingerprint(self) -> str:
        """Changes iff the underlying data (or how it is read) changed."""
        ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class Source:
    """One named input of a pipeline.

    name
        The frame's name inside the ``DataPackage`` and the staging area.
    reader
        Where the raw data comes from. Any ``Reader``.
    spec
        A polspec ``FrameSpec`` subclass or ``TableSpec``. Needed for hermetic
        runs (it is what gets generated) and used to conform real data.
    prepare
        Cleanup applied to the raw frame before conforming. A callable or a
        list of them, applied in order. See ``polars_pipeline.sources.prep``.
    conform
        ``"strict"`` (default): select spec columns and cast, failing on any
        value that will not cast. ``"lenient"``: same, but bad values become
        null. ``"none"``: stage whatever ``prepare`` produced.
    validate_on_ingress
        Run ``polspec.validate`` on the conformed frame before staging.
    synthetic_rows
        Row count for hermetic generation when ``RunOptions`` does not say.
    hermetic
        What a hermetic run does with this source. ``"generate"`` (default)
        replaces it with data generated from ``spec``. ``"real"`` reads it as
        normal -- for reference data that lives in the codebase (lookup CSVs,
        code mappings): always available, deterministic, and better real than
        faked. With a ``spec`` it also serves as a foreign-key parent for the
        generated sources, so synthetic rows reference real codes.
    """

    name: str
    reader: Reader
    spec: Any = None
    prepare: Prepare | Sequence[Prepare] | None = None
    conform: Conform = "strict"
    validate_on_ingress: bool = False
    synthetic_rows: int = 1_000
    hermetic: Literal["generate", "real"] = "generate"

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise SourceError(f"invalid source name {self.name!r}")
        if self.hermetic not in ("generate", "real"):
            raise SourceError(
                f"source {self.name!r}: hermetic must be 'generate' or 'real', "
                f"got {self.hermetic!r}"
            )
        if not isinstance(self.reader, Reader):
            raise SourceError(
                f"source {self.name!r}: reader must implement scan(), fingerprint() "
                f"and describe(); got {type(self.reader).__name__}"
            )
        if self.spec is not None:
            object.__setattr__(self, "spec", as_table_spec(self.spec))
        object.__setattr__(self, "prepare", as_prepare_list(self.prepare))

    # -- the ingress path ----------------------------------------------------

    @property
    def table_spec(self) -> TableSpec | None:
        return self.spec

    def read(self) -> pl.LazyFrame:
        """read -> prepare -> conform. The frame that gets staged."""
        try:
            lf = self.reader.scan()
        except SourceError:
            raise
        except Exception as exc:
            raise SourceError(f"source {self.name!r}: read failed: {exc}") from exc
        lf = self.apply_prepare(lf)
        return self.conform_frame(lf)

    def apply_prepare(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for i, step in enumerate(self.prepare):
            try:
                lf = step(lf)
            except Exception as exc:
                raise SourceError(
                    f"source {self.name!r}: prepare step {i} ({_describe_callable(step)}) "
                    f"failed: {exc}"
                ) from exc
        return lf

    def conform_frame(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Select the spec's columns in order and cast to its dtypes."""
        if self.spec is None or self.conform == "none":
            return lf
        schema = self.spec.schema()
        have = lf.collect_schema()
        missing = [c for c in schema if c not in have]
        if missing:
            raise SourceError(
                f"source {self.name!r}: columns required by spec {self.spec.name!r} are "
                f"missing after prepare: {missing}; present: {have.names()}"
            )
        strict = self.conform == "strict"
        lf = lf.select([pl.col(c).cast(t, strict=strict) for c, t in schema.items()])
        if self.validate_on_ingress:
            lf = polspec_validate(self.spec, lf)
        return lf

    # -- change detection ----------------------------------------------------

    def prepare_hash(self) -> str:
        """Hash of the prepare/conform configuration, so a code change re-stages.

        Uses each callable's source when ``inspect`` can find it, otherwise its
        repr (dataclass helpers in ``prep`` have stable reprs).
        """
        h = hashlib.sha256()
        h.update(f"conform={self.conform};validate={self.validate_on_ingress};".encode())
        if self.spec is not None:
            h.update(repr(self.spec.schema()).encode())
        for step in self.prepare:
            h.update(_callable_identity(step).encode())
            h.update(b"\0")
        return h.hexdigest()


def _describe_callable(fn: Any) -> str:
    return getattr(fn, "__qualname__", None) or repr(fn)


def _callable_identity(fn: Any) -> str:
    """A string that changes when the callable's behaviour plausibly changes."""
    custom = getattr(fn, "__prepare_hash__", None)
    if callable(custom):
        return str(custom())
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        try:
            return f"{fn.__module__}.{fn.__qualname__}\n{inspect.getsource(fn)}"
        except OSError, TypeError:
            return f"{fn.__module__}.{fn.__qualname__}"
    # Callable instances (the dataclass helpers in prep.py): repr carries the config.
    return f"{type(fn).__module__}.{type(fn).__qualname__}:{fn!r}"


__all__ = ["Conform", "FileReader", "FunctionReader", "Prepare", "Reader", "Source"]
