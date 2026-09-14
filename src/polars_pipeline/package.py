"""``DataPackage``: the thing a pipeline runs on and hands back.

It is the union of *configuration*, *where the data comes from* (sources), and
*the frames themselves* -- always lazy inside the package. Two ways to make one:

- ``DataPackage.from_sources(config, sources)``: nothing is read yet; the
  runner's ingress stage stages each source and fills ``frames``.
- ``DataPackage.from_frames(config, {...})``: pre-prepared frames, no ingress.
  Eager frames are wrapped ``.lazy()`` so steps see one type.

In eager (debug) mode the runner also fills ``materialized`` with the
collected ``DataFrame`` after every step, so intermediates can be inspected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from polars_pipeline.config import PipelineConfig
from polars_pipeline.errors import PipelineError
from polars_pipeline.sources import Source
from polars_pipeline.staging import StagingArea

Provenance = Literal["real", "synthetic", "injected", "derived", "supplied"]


@dataclass
class DataPackage:
    config: PipelineConfig
    sources: dict[str, Source] = field(default_factory=dict)
    frames: dict[str, pl.LazyFrame] = field(default_factory=dict)
    staging: StagingArea | None = None
    materialized: dict[str, pl.DataFrame] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_sources(
        cls,
        config: PipelineConfig,
        sources: Iterable[Source],
        *,
        staging: StagingArea | None = None,
    ) -> DataPackage:
        by_name: dict[str, Source] = {}
        for s in sources:
            if s.name in by_name:
                raise PipelineError(f"duplicate source name {s.name!r}")
            by_name[s.name] = s
        return cls(
            config=config,
            sources=by_name,
            staging=staging or StagingArea(config.staging_dir),
        )

    @classmethod
    def from_frames(
        cls,
        config: PipelineConfig,
        frames: Mapping[str, pl.DataFrame | pl.LazyFrame],
        *,
        sources: Iterable[Source] = (),
    ) -> DataPackage:
        pkg = cls(config=config, sources={s.name: s for s in sources})
        for name, frame in frames.items():
            pkg.put(name, frame, provenance="supplied")
        return pkg

    # -- frames --------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self.frames

    def __getitem__(self, name: str) -> pl.LazyFrame:
        return self.get(name)

    def names(self) -> list[str]:
        return list(self.frames)

    def get(self, name: str) -> pl.LazyFrame:
        try:
            return self.frames[name]
        except KeyError:
            have = ", ".join(self.frames) or "nothing"
            raise PipelineError(f"no frame named {name!r} in package (has: {have})") from None

    def put(
        self,
        name: str,
        frame: pl.DataFrame | pl.LazyFrame,
        *,
        provenance: Provenance = "derived",
    ) -> None:
        if isinstance(frame, pl.DataFrame):
            self.materialized[name] = frame
            self.frames[name] = frame.lazy()
        elif isinstance(frame, pl.LazyFrame):
            self.materialized.pop(name, None)
            self.frames[name] = frame
        else:
            raise PipelineError(
                f"frame {name!r} must be a polars DataFrame or LazyFrame, "
                f"got {type(frame).__name__}"
            )
        self.provenance[name] = provenance

    def collect(self, name: str) -> pl.DataFrame:
        """The frame as a DataFrame, cached in ``materialized``."""
        if name not in self.materialized:
            self.materialized[name] = self.get(name).collect()
        return self.materialized[name]

    def collect_all(self, names: Iterable[str] | None = None) -> dict[str, pl.DataFrame]:
        wanted = list(names) if names is not None else self.names()
        pending = [n for n in wanted if n not in self.materialized]
        if pending:
            for n, df in zip(
                pending, pl.collect_all([self.frames[n] for n in pending]), strict=True
            ):
                self.materialized[n] = df
        return {n: self.materialized[n] for n in wanted}

    def schema(self, name: str) -> pl.Schema:
        return self.get(name).collect_schema()

    def __repr__(self) -> str:
        parts = [f"{n} ({self.provenance.get(n, '?')})" for n in self.frames]
        return f"DataPackage(frames=[{', '.join(parts)}])"
