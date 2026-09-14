"""``BasePipeline``: subclass it, declare sources, decorate steps, call ``run``.

::

    class Sales(BasePipeline):
        name = "sales"
        sources = (
            Source("orders", FileReader(r"\\\\nas\\sales\\orders.xlsx", header_row="auto"),
                   spec=OrdersSpec, prepare=[prep.normalise_names()]),
            Source("customers", FileReader("customers.parquet"), spec=CustomersSpec),
        )

        @step(inputs=("orders", "customers"))
        def enriched(self, orders, customers):
            return orders.join(customers, on="customer_id")

        @step("egress")
        def write(self, enriched):
            enriched.sink_parquet(self.config.params["out"])

    Sales(PipelineConfig(staging_dir=".staging")).run(RunOptions(mode="eager"))

Child pipelines are declared in ``children``; their sources and steps are
flattened into the parent under a dotted prefix, so one DAG covers the whole
tree and stage selection, refresh, hermetic generation and eager debugging
behave the same at any depth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from polars_pipeline.config import PipelineConfig, RunOptions, RunResult
from polars_pipeline.errors import DefinitionError
from polars_pipeline.package import DataPackage
from polars_pipeline.sources import Source
from polars_pipeline.step import StepSpec, collect_steps, order_steps

DEFAULT_STAGES: tuple[str, ...] = ("ingress", "transform", "egress")
INGRESS = "ingress"


@dataclass(frozen=True)
class Child:
    """A child pipeline plus how the parent feeds it.

    ``inputs`` maps a name the child expects (one of its ``requires``, or a
    source it would otherwise ingest itself) to a frame name in the parent.
    A mapped source is *not* ingested by the child; the parent's frame is used.
    """

    pipeline: type[BasePipeline]
    inputs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundStep:
    """A flattened step together with the pipeline instance it runs on."""

    spec: StepSpec
    instance: BasePipeline

    @property
    def name(self) -> str:
        return self.spec.qualified


@dataclass(frozen=True)
class Plan:
    """The flattened, validated view of a pipeline tree."""

    stages: tuple[str, ...]
    sources: tuple[Source, ...]
    requires: tuple[str, ...]
    steps: tuple[BoundStep, ...]

    def source(self, name: str) -> Source:
        for s in self.sources:
            if s.name == name:
                return s
        raise KeyError(name)

    def producer_of(self, frame: str) -> BoundStep | None:
        for b in self.steps:
            if frame in b.spec.outputs:
                return b
        return None

    def terminal_outputs(self) -> tuple[str, ...]:
        consumed = {i for b in self.steps for i in b.spec.inputs}
        return tuple(o for b in self.steps for o in b.spec.outputs if o not in consumed)

    def describe(self) -> str:
        lines = [f"stages: {' -> '.join(self.stages)}"]
        if self.requires:
            lines.append(f"requires (supplied externally): {', '.join(self.requires)}")
        lines.append("sources:")
        for s in self.sources:
            spec = f" spec={s.spec.name}" if s.spec is not None else ""
            lines.append(f"  {s.name}: {s.reader.describe()}{spec}")
        lines.append("steps:")
        for b in self.steps:
            s = b.spec
            lines.append(
                f"  [{s.stage}] {b.name}: {', '.join(s.inputs) or '-'} -> "
                f"{', '.join(s.outputs) or '-'}"
            )
        return "\n".join(lines)


class BasePipeline:
    """Base class for every pipeline. See the module docstring for the shape."""

    name: ClassVar[str] = ""
    stages: ClassVar[Sequence[str]] = DEFAULT_STAGES
    sources: ClassVar[Sequence[Source]] = ()
    requires: ClassVar[Sequence[str]] = ()
    children: ClassVar[Mapping[str, type[BasePipeline] | Child]] = {}

    _steps: ClassVar[tuple[StepSpec, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__
        cls.stages = tuple(cls.stages)
        if INGRESS not in cls.stages:
            raise DefinitionError(f"{cls.name}: stages must include {INGRESS!r}")
        names = [s.name for s in cls.sources]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise DefinitionError(f"{cls.name}: duplicate source names {dupes}")
        for prefix in cls.children:
            if not prefix or "." in prefix:
                raise DefinitionError(f"{cls.name}: bad child prefix {prefix!r}")
        cls._steps = tuple(collect_steps(cls))
        # Validate this class on its own; the flattened tree is validated in plan().
        cls.plan_for(config=None)

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._plan = type(self).plan_for(self.config, instance=self)

    # -- plan ----------------------------------------------------------------

    @property
    def plan(self) -> Plan:
        return self._plan

    @classmethod
    def plan_for(
        cls,
        config: PipelineConfig | None,
        *,
        instance: BasePipeline | None = None,
        prefix: str = "",
        fed: Mapping[str, str] | None = None,
    ) -> Plan:
        """Flatten this pipeline and its children into one validated ``Plan``.

        Without ``instance`` the steps are bound to a throwaway instance; that
        is only used for class-time validation, never to run.
        """
        fed = dict(fed or {})
        owner = instance if instance is not None else cls.__new__(cls)
        if instance is None:
            owner.config = config or PipelineConfig()

        def pfx(n: str) -> str:
            return f"{prefix}.{n}" if prefix else n

        sources: list[Source] = []
        for s in cls.sources:
            if s.name in fed:
                continue  # parent supplies it
            sources.append(replace(s, name=pfx(s.name)))
        requires = [pfx(r) for r in cls.requires if r not in fed]
        steps: list[BoundStep] = []
        for spec in cls._steps:
            bound = spec if not prefix else spec.prefixed(prefix, fed)
            if prefix:
                bound = replace(bound, owner=prefix)
            elif fed:
                bound = replace(bound, inputs=tuple(fed.get(i, i) for i in bound.inputs))
            steps.append(BoundStep(bound, owner))

        stages = list(cls.stages)
        for child_prefix, child in cls.children.items():
            ref = child if isinstance(child, Child) else Child(child)
            child_cls = ref.pipeline
            child_instance = child_cls(owner.config) if instance is not None else None
            sub = child_cls.plan_for(
                owner.config,
                instance=child_instance,
                prefix=pfx(child_prefix),
                fed={k: v for k, v in ref.inputs.items()},
            )
            for st in sub.stages:
                if st not in stages:
                    stages.append(st)
            sources.extend(sub.sources)
            requires.extend(sub.requires)
            steps.extend(sub.steps)

        # Frames the parent feeds in count as external here; the parent's own
        # flattened pass checks that they really exist.
        external = [s.name for s in sources] + requires + list(fed.values())
        ordered = order_steps(
            [b.spec for b in steps],
            stages=stages,
            produced_externally=external,
            context=cls.name if not prefix else f"{cls.name} (as {prefix!r})",
        )
        by_spec = {id(b.spec): b for b in steps}
        return Plan(
            stages=tuple(stages),
            sources=tuple(sources),
            requires=tuple(requires),
            steps=tuple(by_spec[id(s)] for s in ordered),
        )

    # -- running -------------------------------------------------------------

    def package(self) -> DataPackage:
        """A fresh, un-ingested package for this pipeline's sources."""
        return DataPackage.from_sources(self.config, self.plan.sources)

    def run(
        self,
        options: RunOptions | DataPackage | None = None,
        /,
        *,
        package: DataPackage | None = None,
        **option_kwargs: Any,
    ) -> RunResult:
        """Run the pipeline.

        ``run(RunOptions(...))``, ``run(mode="eager", stages=["transform"])``,
        or ``run(package=DataPackage.from_frames(...))``. See ``RunOptions``.
        """
        from polars_pipeline.runner import Runner

        if isinstance(options, DataPackage):
            package, options = options, None
        if options is None:
            options = RunOptions(**option_kwargs)
        elif option_kwargs:
            options = replace(options, **option_kwargs)
        return Runner(self, package or self.package(), options).run()

    def explain(self) -> str:
        return self.plan.describe()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, steps={len(self.plan.steps)})"
