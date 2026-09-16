"""The staging area: every source lands here once as Parquet, then is scanned.

Layout::

    <root>/
      manifest.json
      <source>.parquet
      hermetic/<seed>/            # same layout, for synthetic runs

The manifest records, per staged source, the reader fingerprint and the
prepare hash at the time of staging, so a later run can tell whether the
staged copy is still current.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from polars_pipeline.errors import StagingError
from polars_pipeline.sources import Source

Provenance = Literal["real", "synthetic", "injected"]

MANIFEST = "manifest.json"


@dataclass
class StagedEntry:
    name: str
    file: str
    fingerprint: str | None
    prepare_hash: str | None
    provenance: Provenance
    staged_at: str
    rows: int | None
    schema: dict[str, str]
    seed: int | None = None
    reader: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class StagingArea:
    """A directory of staged Parquet files plus a manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._entries: dict[str, StagedEntry] | None = None

    # -- variants ------------------------------------------------------------

    def hermetic(self, seed: int | None) -> StagingArea:
        """The sibling area synthetic data for ``seed`` is staged into."""
        leaf = str(seed) if seed is not None else "unseeded"
        return StagingArea(self.root / "hermetic" / leaf)

    # -- manifest ------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST

    def _load(self) -> dict[str, StagedEntry]:
        if self._entries is None:
            self._entries = {}
            if self.manifest_path.exists():
                try:
                    raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise StagingError(f"corrupt manifest at {self.manifest_path}: {exc}") from exc
                for name, data in raw.get("sources", {}).items():
                    self._entries[name] = StagedEntry(**data)
        return self._entries

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "sources": {k: asdict(v) for k, v in self._load().items()}}
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def reload(self) -> None:
        self._entries = None

    # -- queries -------------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._load())

    def has(self, name: str) -> bool:
        entry = self._load().get(name)
        return entry is not None and (self.root / entry.file).exists()

    def entry(self, name: str) -> StagedEntry:
        try:
            return self._load()[name]
        except KeyError:
            raise StagingError(f"{name!r} is not staged in {self.root}") from None

    def path_for(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def is_stale(self, source: Source, *, fingerprint: str | None = None) -> bool:
        """True when the staged copy is not what reading ``source`` would give.

        A missing entry is stale. So is a synthetic or injected entry: it was
        never read from the source, so a real ingress must replace it (hermetic
        reuse is decided by seed in the runner, not here). Pass ``fingerprint``
        when the caller has already computed it, to avoid a second probe.
        """
        if not self.has(source.name):
            return True
        entry = self.entry(source.name)
        if entry.provenance != "real":
            return True
        if fingerprint is None:
            fingerprint = source.reader.fingerprint()
        return entry.fingerprint != fingerprint or entry.prepare_hash != source.prepare_hash()

    # -- read / write --------------------------------------------------------

    def scan(self, name: str) -> pl.LazyFrame:
        if not self.has(name):
            raise StagingError(
                f"{name!r} is not staged in {self.root}; run the ingress stage first"
            )
        return pl.scan_parquet(self.root / self.entry(name).file)

    def stage(
        self,
        source: Source,
        lf: pl.LazyFrame | pl.DataFrame,
        *,
        provenance: Provenance = "real",
        seed: int | None = None,
        fingerprint: str | None = None,
        prepare_hash: str | None = None,
    ) -> StagedEntry:
        """Write ``lf`` as ``<name>.parquet`` and record it in the manifest.

        Writes go to a temp file in the same directory and are renamed into
        place, so a crash mid-write never leaves a half-staged file behind
        with a manifest entry pointing at it.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(source.name)
        tmp = self.root / f".{source.name}.tmp.parquet"
        _unlink_retry(tmp)  # leftover from an earlier crash
        try:
            rows = _write_parquet(lf, tmp)
            tmp.replace(target)
        except Exception as exc:
            _unlink_retry(tmp, linger=0.2)
            raise StagingError(f"staging {source.name!r} failed: {exc}") from exc

        schema = pl.read_parquet_schema(target)
        entry = StagedEntry(
            name=source.name,
            file=target.name,
            fingerprint=fingerprint,
            prepare_hash=prepare_hash,
            provenance=provenance,
            staged_at=datetime.now(UTC).isoformat(timespec="seconds"),
            rows=rows,
            schema={k: str(v) for k, v in schema.items()},
            seed=seed,
            reader=source.reader.describe() if provenance == "real" else None,
        )
        self._load()[source.name] = entry
        self._save()
        return entry

    def drop(self, name: str) -> None:
        entries = self._load()
        entry = entries.pop(name, None)
        if entry is not None:
            (self.root / entry.file).unlink(missing_ok=True)
            self._save()

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self._entries = None

    def __repr__(self) -> str:
        return f"StagingArea({str(self.root)!r}, staged={self.names()})"


def _unlink_retry(path: Path, *, linger: float = 0.0) -> None:
    """Best-effort delete.

    After a failed ``sink_parquet`` Polars' writer thread can still create the
    file a few milliseconds *after* the exception reached us, so on the failure
    path we keep deleting for ``linger`` seconds rather than trusting a single
    "does not exist".
    """
    deadline = time.monotonic() + linger
    while True:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass  # next stage() of this source clears it before writing


def _write_parquet(lf: pl.LazyFrame | pl.DataFrame, path: Path) -> int:
    """Write the frame and return its row count.

    ``sink_parquet`` streams every plan shape Polars can build (in-memory
    sources included), so a failure here is a real query error and is left to
    propagate rather than retried in memory.
    """
    if isinstance(lf, pl.DataFrame):
        lf.write_parquet(path)
        return lf.height
    lf.sink_parquet(path)
    return pl.scan_parquet(path).select(pl.len()).collect().item()
