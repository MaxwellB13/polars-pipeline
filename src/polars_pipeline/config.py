"""Run-time configuration and results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl

from polars_pipeline.errors import PipelineError

Mode = Literal["lazy", "eager"]
RefreshPolicy = Literal["none", "stale", "all"] | frozenset[str] | set[str]
Frames = Mapping[str, pl.DataFrame | pl.LazyFrame]


@dataclass(frozen=True)
class PipelineConfig:
    """Static configuration a pipeline is constructed with.

    ``staging_dir`` is where every ingested source lands as Parquet. ``params``
    is a free-form bag for pipeline-specific settings (dates, thresholds...)
    that steps read via ``self.config.params``.
    """

    staging_dir: Path = Path(".staging")
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "staging_dir", Path(self.staging_dir))


@dataclass(frozen=True)
class RunOptions:
    """Everything that varies between two runs of the same pipeline.

    mode
        ``"lazy"`` builds one Polars plan per output and collects nothing;
        ``"eager"`` collects after every step so failures surface where they
        happen and every intermediate is inspectable on the package.
    hermetic
        Replace every source with polspec-generated data. Nothing external is
        read. Requires a ``spec`` on every source.
    seed / synthetic_rows
        Hermetic generation controls. ``synthetic_rows`` overrides each
        source's own default, either globally (int) or per source (mapping).
    stages / steps
        Run only these stages, or only these named steps. Inputs those steps
        need must already be staged. ``steps=`` on its own skips ingress (a
        targeted rerun); pass ``stages=["ingress", ...]`` too to re-ingest.
    refresh
        Which sources to re-ingest: ``"none"`` (only if absent), ``"stale"``
        (fingerprint or prepare changed), ``"all"``, or a set of names.
    references
        Frames to inject for named sources instead of reading or generating
        them. Useful for pinning one real table inside a hermetic run, or for
        hand-built edge-case inputs.
    """

    mode: Mode = "lazy"
    hermetic: bool = False
    seed: int | None = None
    synthetic_rows: int | Mapping[str, int] | None = None
    stages: Sequence[str] | None = None
    steps: Sequence[str] | None = None
    refresh: RefreshPolicy = "stale"
    references: Frames | None = None

    def __post_init__(self) -> None:
        # A bare string is a Sequence[str] of its characters; catch the slip.
        for field_name in ("stages", "steps"):
            value = getattr(self, field_name)
            if isinstance(value, str):
                raise PipelineError(
                    f"{field_name}= takes a list of names, not a string; "
                    f"use {field_name}=[{value!r}]"
                )
        if isinstance(self.refresh, str):
            if self.refresh not in ("none", "stale", "all"):
                raise PipelineError(
                    "refresh= must be 'none', 'stale', 'all' or a set of source names, "
                    f"got {self.refresh!r}"
                )
        else:
            object.__setattr__(self, "refresh", frozenset(self.refresh))

    def rows_for(self, name: str, default: int) -> int:
        if self.synthetic_rows is None:
            return default
        if isinstance(self.synthetic_rows, int):
            return self.synthetic_rows
        return int(self.synthetic_rows.get(name, default))

    def wants_refresh(self, name: str, *, stale: bool, present: bool) -> bool:
        if not present:
            return True
        policy = self.refresh
        if policy == "none":
            return False
        if policy == "all":
            return True
        if policy == "stale":
            return stale
        return name in policy


@dataclass
class StepReport:
    name: str
    stage: str
    status: Literal["ran", "skipped", "failed"]
    seconds: float = 0.0
    error: str | None = None


@dataclass
class RunResult:
    """What a run did. ``package`` holds the frames; the rest is bookkeeping."""

    package: Any  # DataPackage; typed loosely to avoid an import cycle
    options: RunOptions
    steps: list[StepReport] = field(default_factory=list)
    ingested: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)

    @property
    def ran(self) -> list[str]:
        """Names of the steps that ran (ingress is reported via ``ingested``)."""
        return [s.name for s in self.steps if s.status == "ran" and s.stage != "ingress"]

    @property
    def skipped(self) -> list[str]:
        return [s.name for s in self.steps if s.status == "skipped" and s.stage != "ingress"]

    @property
    def failed(self) -> str | None:
        return next((s.name for s in self.steps if s.status == "failed"), None)

    def summary(self) -> str:
        lines = [
            f"mode={self.options.mode} hermetic={self.options.hermetic} seed={self.options.seed}",
            f"ingested: {', '.join(self.ingested) or '-'}",
            f"reused:   {', '.join(self.reused) or '-'}",
        ]
        for s in self.steps:
            tail = f" ({s.seconds:.3f}s)" if s.status == "ran" else ""
            err = f" -> {s.error}" if s.error else ""
            lines.append(f"  [{s.stage}] {s.name}: {s.status}{tail}{err}")
        return "\n".join(lines)
