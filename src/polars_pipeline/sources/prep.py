"""Composable ``LazyFrame -> LazyFrame`` helpers for ``Source.prepare``.

Each helper is a small callable you can list in ``Source(prepare=[...])``.
They exist because raw files -- Excel especially -- rarely arrive stage-ready:
banner rows above the header, trailing notes, column names with units and
whitespace. Fix those here so the staged Parquet already matches the spec.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

Prepare = Callable[[pl.LazyFrame], pl.LazyFrame]

_NON_WORD = re.compile(r"[^0-9a-zA-Z]+")


def _snake(name: str) -> str:
    name = _NON_WORD.sub("_", name.strip()).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    if name and name[0].isdigit():
        name = f"_{name}"
    return name or "unnamed"


def _dedupe(names: Iterable[str]) -> list[str]:
    """Suffix repeats with ``_2``, ``_3``... so a frame never gets two equal names."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        n = seen.get(name, 0) + 1
        seen[name] = n
        out.append(name if n == 1 else f"{name}_{n}")
    return out


@dataclass(frozen=True)
class normalise_names:
    """Snake-case every column name; de-duplicate collisions with ``_2``, ``_3``...

    Names starting with ``__`` (the reader's own markers, e.g. ``__sheet``) are
    left alone.
    """

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        cols = [c for c in lf.collect_schema().names() if not c.startswith("__")]
        return lf.rename(dict(zip(cols, _dedupe(_snake(c) for c in cols), strict=True)))


@dataclass(frozen=True)
class rename:
    mapping: Mapping[str, str]

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.rename(dict(self.mapping))


@dataclass(frozen=True)
class select_columns:
    columns: tuple[str, ...]

    def __init__(self, *columns: str) -> None:
        object.__setattr__(self, "columns", tuple(columns))

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select(list(self.columns))


@dataclass(frozen=True)
class drop_empty_rows:
    """Drop rows where every column (or every column in ``subset``) is null."""

    subset: tuple[str, ...] | None = None

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        cols = list(self.subset) if self.subset else lf.collect_schema().names()
        return lf.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols]))


@dataclass(frozen=True)
class drop_empty_columns:
    """Drop columns that are entirely null. Collects a null-count pass."""

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        counts = lf.null_count().collect()
        height = lf.select(pl.len()).collect().item()
        keep = [c for c in counts.columns if counts[c].item() < height]
        return lf.select(keep)


@dataclass(frozen=True)
class promote_header:
    """Use a data row as the header, dropping everything at and above it.

    ``row="auto"`` picks the first row that has no nulls at all -- in practice the
    real header of a report with banner rows above it. ``row=n`` uses the n-th row
    (0-based). Works on any frame, so it's engine-agnostic: reads the head only.
    """

    row: int | Literal["auto"] = "auto"
    scan_rows: int = 50

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        head = lf.head(self.scan_rows).collect()
        if self.row == "auto":
            idx = _first_full_row(head)
        else:
            idx = int(self.row)
        names = _dedupe(_clean_header(v, i) for i, v in enumerate(head.row(idx)))
        out = lf.slice(idx + 1)
        return out.rename(dict(zip(out.collect_schema().names(), names, strict=True)))


def _first_full_row(df: pl.DataFrame) -> int:
    if df.height == 0:
        raise ValueError("promote_header: frame is empty")
    full = df.select(pl.all_horizontal([pl.col(c).is_not_null() for c in df.columns]))
    for i, ok in enumerate(full.to_series()):
        if ok:
            return i
    # No row is fully populated: fall back to the row with the most non-nulls.
    counts = df.select(pl.sum_horizontal([pl.col(c).is_not_null() for c in df.columns]))
    return int(counts.to_series().arg_max() or 0)


def _clean_header(value: object, index: int) -> str:
    if value is None:
        return f"column_{index + 1}"
    text = str(value).strip()
    return text or f"column_{index + 1}"


@dataclass(frozen=True)
class cast:
    mapping: Mapping[str, pl.DataType]
    strict: bool = True

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            [pl.col(c).cast(t, strict=self.strict) for c, t in self.mapping.items()]
        )


@dataclass(frozen=True)
class strip_strings:
    """Trim whitespace on every String column (or on ``columns``)."""

    columns: tuple[str, ...] | None = None

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.columns:
            return lf.with_columns([pl.col(c).str.strip_chars() for c in self.columns])
        return lf.with_columns(pl.col(pl.String).str.strip_chars())


@dataclass(frozen=True)
class filter_rows:
    predicate: pl.Expr

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(self.predicate)


def compose(steps: Iterable[Prepare]) -> Prepare:
    steps = tuple(steps)

    def run(lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in steps:
            lf = step(lf)
        return lf

    return run


def as_prepare_list(prepare: Prepare | Sequence[Prepare] | None) -> tuple[Prepare, ...]:
    if prepare is None:
        return ()
    if callable(prepare):
        return (prepare,)
    return tuple(prepare)
