"""Executes a ``Plan`` against a ``DataPackage`` under some ``RunOptions``.

Ingress (per source, in declaration order):

    injected via options.references -> stage as "injected"
    hermetic                        -> polspec Registry.generate_all -> stage as "synthetic"
    otherwise                       -> refresh policy decides: reuse staged copy,
                                       or read -> prepare -> conform -> stage

Then the selected steps run in plan order. Lazy mode passes plans through;
eager mode collects every output as it is produced, so an error is raised by
the step that caused it and every intermediate is on ``package.materialized``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import polars as pl
from polspec import Registry
from polspec import validate as polspec_validate
from polspec.errors import PolspecError

from polars_pipeline.config import RunOptions, RunResult, StepReport
from polars_pipeline.errors import HermeticError, MissingInputError, StepError
from polars_pipeline.package import DataPackage
from polars_pipeline.pipeline import INGRESS, BoundStep, Plan
from polars_pipeline.sources import Source
from polars_pipeline.staging import StagingArea

if TYPE_CHECKING:
    from polars_pipeline.pipeline import BasePipeline


class Runner:
    def __init__(self, pipeline: BasePipeline, package: DataPackage, options: RunOptions) -> None:
        self.pipeline = pipeline
        self.plan: Plan = pipeline.plan
        self.package = package
        self.options = options
        self.result = RunResult(package=package, options=options)
        self._validate_options()

        # A package built from frames has no sources; adopt the plan's so that
        # partial runs and hermetic generation still know what exists.
        for s in self.plan.sources:
            package.sources.setdefault(s.name, s)

        base = package.staging or StagingArea(package.config.staging_dir)
        package.staging = base
        self.staging: StagingArea = base.hermetic(options.seed) if options.hermetic else base

    # -- entry ---------------------------------------------------------------

    def run(self) -> RunResult:
        selected_stages = self._selected_stages()
        # steps=[...] on its own is a targeted rerun and skips ingress; naming
        # "ingress" in stages= explicitly brings it back.
        run_ingress = INGRESS in selected_stages and (
            self.options.steps is None or self.options.stages is not None
        )
        if run_ingress:
            self._ingress()
        else:
            for s in self.plan.sources:
                self.result.steps.append(StepReport(s.name, INGRESS, "skipped"))

        selected = self._selected_steps(selected_stages)
        self._ensure_inputs(selected)
        for bound in self.plan.steps:
            if bound in selected:
                self._run_step(bound)
            else:
                self.result.steps.append(StepReport(bound.name, bound.spec.stage, "skipped"))
        return self.result

    # -- selection -----------------------------------------------------------

    def _validate_options(self) -> None:
        o = self.options
        known_stages = set(self.plan.stages)
        if o.stages is not None:
            unknown = [s for s in o.stages if s not in known_stages]
            if unknown:
                raise MissingInputError(
                    f"unknown stage(s) {unknown}; this pipeline has {list(self.plan.stages)}"
                )
        if o.steps is not None:
            names = {b.name for b in self.plan.steps}
            unknown = [s for s in o.steps if s not in names]
            if unknown:
                raise MissingInputError(f"unknown step(s) {unknown}; known: {sorted(names)}")
        if o.references:
            names = {s.name for s in self.plan.sources} | set(self.plan.requires)
            unknown = [r for r in o.references if r not in names]
            if unknown:
                raise MissingInputError(
                    f"references given for unknown sources {unknown}; known: {sorted(names)}"
                )

    def _selected_stages(self) -> set[str]:
        if self.options.stages is None:
            return set(self.plan.stages)
        return set(self.options.stages)

    def _selected_steps(self, stages: set[str]) -> set[BoundStep]:
        chosen = set()
        for b in self.plan.steps:
            if b.spec.stage not in stages:
                continue
            if self.options.steps is not None and b.name not in self.options.steps:
                continue
            chosen.add(b)
        return chosen

    # -- ingress -------------------------------------------------------------

    def _ingress(self) -> None:
        pkg = self.package
        # Frames already in the package (DataPackage.from_frames) stand in for
        # their sources exactly like options.references.
        refs: dict[str, Any] = {
            name: pkg.frames[name]
            for name in pkg.sources
            if name in pkg.frames and pkg.provenance.get(name) == "supplied"
        }
        refs.update(self.options.references or {})

        # Frames supplied by the caller, real or hermetic alike.
        for name, frame in refs.items():
            if name not in pkg.sources:
                pkg.put(name, frame, provenance="injected")
                continue
            src = pkg.sources[name]
            lf = src.conform_frame(frame.lazy() if isinstance(frame, pl.DataFrame) else frame)
            self.staging.stage(src, lf, provenance="injected", seed=self.options.seed)
            self._adopt(name, "injected")

        pending = [s for s in self.plan.sources if s.name not in refs]
        if self.options.hermetic:
            self._ingress_hermetic(pending, refs)
        else:
            for src in pending:
                self._ingress_real(src)

    def _ingress_real(self, src: Source) -> None:
        present = self.staging.has(src.name)
        stale = present and self.staging.is_stale(src)
        if not self.options.wants_refresh(src.name, stale=stale, present=present):
            self._adopt(src.name, "real", reused=True)
            return
        t0 = time.perf_counter()
        self.staging.stage(
            src,
            src.read(),
            provenance="real",
            fingerprint=src.reader.fingerprint(),
            prepare_hash=src.prepare_hash(),
        )
        self._adopt(src.name, "real", seconds=time.perf_counter() - t0)

    def _ingress_hermetic(self, pending: list[Source], refs: Mapping[str, Any]) -> None:
        missing = [s.name for s in pending if s.spec is None]
        if missing:
            raise HermeticError(f"hermetic run needs a spec on every source; missing for {missing}")
        # Generate only what the refresh policy asks for; reuse the rest by seed.
        # Without a seed there is nothing stable to reuse, so always generate.
        seeded = self.options.seed is not None
        to_generate = [
            s
            for s in pending
            if self.options.wants_refresh(
                s.name, stale=False, present=seeded and self.staging.has(s.name)
            )
        ]
        generating = {s.name for s in to_generate}
        for s in pending:
            if s.name not in generating:
                self._adopt(s.name, "synthetic", reused=True)
        if not to_generate:
            return

        registry = Registry()
        spec_to_sources: dict[str, list[Source]] = {}
        rows: dict[str, int] = {}
        for s in to_generate:
            assert s.spec is not None
            registry.add(s.spec)
            spec_to_sources.setdefault(s.spec.name, []).append(s)
            rows[s.spec.name] = max(
                rows.get(s.spec.name, 0), self.options.rows_for(s.name, s.synthetic_rows)
            )
        # Parents already staged (reused or injected) satisfy foreign keys.
        references: dict[str, pl.DataFrame | pl.LazyFrame] = {}
        for s in pending:
            if s.spec is not None and s.name not in generating and self.staging.has(s.name):
                references.setdefault(s.spec.name, self.staging.scan(s.name))
        for name, frame in refs.items():
            src = self.package.sources.get(name)
            if src is not None and src.spec is not None:
                references.setdefault(src.spec.name, frame)

        t0 = time.perf_counter()
        try:
            generated = registry.generate_all(rows, seed=self.options.seed, references=references)
        except PolspecError as exc:
            raise HermeticError(f"synthetic generation failed: {exc}") from exc
        for spec_name, sources in spec_to_sources.items():
            df = generated[spec_name]
            for s in sources:
                self.staging.stage(
                    s, s.conform_frame(df.lazy()), provenance="synthetic", seed=self.options.seed
                )
                self._adopt(s.name, "synthetic", seconds=time.perf_counter() - t0)

    def _adopt(
        self, name: str, provenance: str, *, reused: bool = False, seconds: float = 0.0
    ) -> None:
        self.package.put(name, self.staging.scan(name), provenance=provenance)  # type: ignore[arg-type]
        (self.result.reused if reused else self.result.ingested).append(name)
        self.result.steps.append(
            StepReport(name, INGRESS, "skipped" if reused else "ran", seconds=seconds)
        )

    # -- steps ---------------------------------------------------------------

    def _ensure_inputs(self, selected: set[BoundStep]) -> None:
        """Every input a selected step needs is in the package or loadable from staging."""
        produced = {o for b in selected for o in b.spec.outputs}
        for b in self.plan.steps:
            if b not in selected:
                continue
            for inp in b.spec.inputs:
                if inp in self.package or inp in produced:
                    continue
                if self.staging.has(inp):
                    prov = self.staging.entry(inp).provenance
                    self.package.put(inp, self.staging.scan(inp), provenance=prov)
                    self.result.reused.append(inp)
                    continue
                producer = self.plan.producer_of(inp)
                hint = (
                    f"it is produced by step {producer.name!r} in stage {producer.spec.stage!r}"
                    if producer
                    else f"it is a source; run the {INGRESS!r} stage first"
                    if inp in self.package.sources
                    else "it must be supplied in the package (requires=)"
                )
                raise MissingInputError(
                    f"step {b.name!r} needs {inp!r}, which is not available: {hint}"
                )

    def _run_step(self, bound: BoundStep) -> None:
        spec = bound.spec
        args = [self.package.get(i) for i in spec.inputs]
        t0 = time.perf_counter()
        try:
            out = spec.fn(bound.instance, *args)
            outputs = self._unpack(bound, out)
            for name, lf in outputs.items():
                if spec.validate is not None:
                    lf = polspec_validate(spec.validate, lf)
                if self.options.mode == "eager":
                    df = lf.collect()
                    self.package.put(name, df)
                else:
                    self.package.put(name, lf)
        except Exception as exc:
            self.result.steps.append(
                StepReport(bound.name, spec.stage, "failed", time.perf_counter() - t0, str(exc))
            )
            raise StepError(bound.name, exc) from exc
        self.result.steps.append(
            StepReport(bound.name, spec.stage, "ran", time.perf_counter() - t0)
        )

    @staticmethod
    def _unpack(bound: BoundStep, out: Any) -> dict[str, pl.LazyFrame]:
        names = bound.spec.outputs
        if not names:
            return {}
        if len(names) == 1 and isinstance(out, pl.DataFrame | pl.LazyFrame):
            frames: list[Any] = [out]
        elif isinstance(out, Mapping):
            missing = [n for n in names if n not in out]
            if missing:
                raise TypeError(f"returned dict lacks outputs {missing}")
            frames = [out[n] for n in names]
        elif isinstance(out, tuple | list):
            if len(out) != len(names):
                raise TypeError(f"returned {len(out)} frames for outputs {list(names)}")
            frames = list(out)
        else:
            raise TypeError(
                f"must return a LazyFrame for outputs {list(names)}, got {type(out).__name__}"
            )
        result: dict[str, pl.LazyFrame] = {}
        for name, f in zip(names, frames, strict=True):
            if isinstance(f, pl.DataFrame):
                f = f.lazy()
            elif not isinstance(f, pl.LazyFrame):
                raise TypeError(f"output {name!r} is {type(f).__name__}, not a frame")
            result[name] = f
        return result
