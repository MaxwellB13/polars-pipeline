"""Reading files on local or UNC paths into a raw LazyFrame."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl

from polars_pipeline.errors import SourceError
from polars_pipeline.sources.prep import promote_header

Format = Literal["auto", "parquet", "csv", "ipc", "ndjson", "excel"]

_SUFFIXES: dict[str, Format] = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".ipc": "ipc",
    ".arrow": "ipc",
    ".feather": "ipc",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xlsb": "excel",
    ".xls": "excel",
}


@dataclass(frozen=True)
class FileReader:
    r"""A file (parquet/csv/ipc/ndjson lazily, excel eagerly) as a ``Reader``.

    ``path`` may be a UNC path (``\\server\share\file.xlsx``); ``Path`` handles
    it. For Excel:

    sheet
        A sheet name or 0-based index, or a list of names: each is read, the
        header handled per sheet, then all are concatenated diagonally with a
        ``__sheet`` column naming the origin.
    header_row
        ``0`` (default) trusts the first row as the header. ``"auto"`` reads
        headerless and promotes the first fully-populated row, dropping banner
        rows above it. ``n`` promotes row ``n`` (0-based, counted from the top
        of the sheet). ``None`` keeps the sheet headerless (``column_1``...)
        for ``prepare`` to deal with.
    read_kwargs
        Passed straight to the underlying ``pl.scan_*`` / ``pl.read_excel``.
    """

    path: Path
    format: Format = "auto"
    sheet: str | int | Sequence[str] | None = None
    header_row: int | Literal["auto"] | None = 0
    read_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if isinstance(self.sheet, list):
            object.__setattr__(self, "sheet", tuple(self.sheet))
        object.__setattr__(self, "read_kwargs", dict(self.read_kwargs))
        if self.format == "auto":
            suffix = self.path.suffix.lower()
            if suffix not in _SUFFIXES:
                raise SourceError(f"cannot infer format from {self.path.name!r}; pass format=...")
            object.__setattr__(self, "format", _SUFFIXES[suffix])

    # -- Reader protocol ---------------------------------------------------

    def describe(self) -> str:
        extra = f" sheet={self.sheet!r}" if self.sheet is not None else ""
        return f"{self.format}:{self.path}{extra}"

    def fingerprint(self) -> str:
        try:
            st = self.path.stat()
        except OSError as exc:
            raise SourceError(f"cannot stat {self.path}: {exc}") from exc
        payload = {
            "path": str(self.path.resolve()),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "format": self.format,
            "sheet": list(self.sheet) if isinstance(self.sheet, tuple) else self.sheet,
            "header_row": self.header_row,
            "read_kwargs": _stable(self.read_kwargs),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def scan(self) -> pl.LazyFrame:
        if not self.path.exists():
            raise SourceError(f"file not found: {self.path}")
        kw = dict(self.read_kwargs)
        match self.format:
            case "parquet":
                return pl.scan_parquet(self.path, **kw)
            case "csv":
                if self.path.suffix.lower() == ".tsv":
                    kw.setdefault("separator", "\t")
                return pl.scan_csv(self.path, **kw)
            case "ipc":
                return pl.scan_ipc(self.path, **kw)
            case "ndjson":
                return pl.scan_ndjson(self.path, **kw)
            case "excel":
                return self._scan_excel(kw)
        raise SourceError(f"unsupported format {self.format!r}")  # pragma: no cover

    # -- Excel -------------------------------------------------------------

    def _scan_excel(self, kw: dict[str, Any]) -> pl.LazyFrame:
        headerless = self.header_row != 0
        kw.setdefault("has_header", not headerless)
        if isinstance(self.sheet, tuple):
            frames = []
            for name in self.sheet:
                df = self._read_sheet(name, kw)
                frames.append(
                    self._apply_header(df.lazy()).with_columns(pl.lit(name).alias("__sheet"))
                )
            return pl.concat(frames, how="diagonal_relaxed")
        df = self._read_sheet(self.sheet, kw)
        return self._apply_header(df.lazy())

    def _read_sheet(self, sheet: str | int | None, kw: dict[str, Any]) -> pl.DataFrame:
        if isinstance(sheet, int):
            kw["sheet_id"] = sheet + 1
        elif isinstance(sheet, str):
            kw["sheet_name"] = sheet
        try:
            with warnings.catch_warnings():
                # polars' own fastexcel bridge trips a FutureWarning; not ours to fix.
                warnings.simplefilter("ignore", FutureWarning)
                return pl.read_excel(self.path, **kw)
        except ImportError as exc:  # fastexcel missing
            raise SourceError(
                "reading Excel needs the 'excel' extra: uv add 'polars-pipeline[excel]'"
            ) from exc

    def _apply_header(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.header_row in (0, None):
            return lf
        return promote_header(row=self.header_row)(lf)


def _stable(obj: Any) -> Any:
    """JSON-able, order-stable view of read kwargs for fingerprinting."""
    if isinstance(obj, Mapping):
        return {str(k): _stable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_stable(v) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return repr(obj)
