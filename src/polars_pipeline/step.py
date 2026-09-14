"""The ``@step`` decorator and the DAG built from decorated methods.

A step declares the frames it reads and the frames it writes. That is all the
runner needs: inputs are pulled from the ``DataPackage`` by name and passed
positionally, the return value is stored under ``outputs``. Steps only ever
see ``LazyFrame``s, whichever mode the pipeline runs in.
"""

from __future__ import annotations

import heapq
import inspect
import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from polspec.tablespec import TableSpec, as_table_spec

from polars_pipeline.errors import DefinitionError

STEP_ATTR = "__pipeline_step__"
_counter = itertools.count()


@dataclass(frozen=True)
class StepSpec:
    """Static description of one step, as declared on a pipeline class."""

    name: str
    stage: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    fn: Callable[..., Any]
    validate: TableSpec | None = None
    order: int = 0
    owner: str = ""  # dotted prefix of the (child) pipeline that declared it

    @property
    def qualified(self) -> str:
        return f"{self.owner}.{self.name}" if self.owner else self.name

    def prefixed(self, prefix: str, rename_inputs: Mapping[str, str] | None = None) -> StepSpec:
        """This step as seen from a parent pipeline that includes its owner."""
        rename_inputs = rename_inputs or {}
        return replace(
            self,
            inputs=tuple(rename_inputs.get(i, f"{prefix}.{i}") for i in self.inputs),
            outputs=tuple(f"{prefix}.{o}" for o in self.outputs),
            owner=f"{prefix}.{self.owner}" if self.owner else prefix,
        )


def step(
    stage: str = "transform",
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | str | None = None,
    validate: Any = None,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a ``BasePipeline`` method as a step.

    stage
        Which stage the step belongs to; must be one of the pipeline's
        ``stages``. Defaults to ``"transform"``.
    inputs
        Frame names to pass in, positionally. Defaults to the method's
        parameter names (after ``self``), so ``def enrich(self, orders,
        customers)`` reads ``orders`` and ``customers``.
    outputs
        Frame name(s) the return value is stored under. Defaults to the
        method name. Return one ``LazyFrame`` for one output, a tuple in
        declared order or a dict keyed by name for several, ``None`` for none
        (an egress step that writes to disk, say).
    validate
        A polspec ``FrameSpec``/``TableSpec`` to validate the (single) output
        against. Lazy in lazy mode, on the collected frame in eager mode.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        step_name = name or fn.__name__
        if inputs is None:
            params = list(inspect.signature(fn).parameters.values())[1:]  # drop self
            resolved_inputs = tuple(
                p.name
                for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
            )
        else:
            resolved_inputs = tuple(inputs)
        if outputs is None:
            resolved_outputs: tuple[str, ...] = (step_name,)
        elif isinstance(outputs, str):
            resolved_outputs = (outputs,)
        else:
            resolved_outputs = tuple(outputs)
        if validate is not None and len(resolved_outputs) != 1:
            raise DefinitionError(
                f"step {step_name!r}: validate= needs exactly one output, got {resolved_outputs}"
            )
        spec = StepSpec(
            name=step_name,
            stage=stage,
            inputs=resolved_inputs,
            outputs=resolved_outputs,
            fn=fn,
            validate=None if validate is None else as_table_spec(validate),
            order=next(_counter),
        )
        setattr(fn, STEP_ATTR, spec)
        return fn

    return decorate


def collect_steps(cls: type) -> list[StepSpec]:
    """Every ``@step`` method on ``cls`` (inherited ones included), in definition order."""
    found: dict[str, StepSpec] = {}
    for klass in reversed(cls.__mro__):
        for attr, value in vars(klass).items():
            spec = getattr(value, STEP_ATTR, None)
            if isinstance(spec, StepSpec):
                found[attr] = spec  # a subclass override replaces the parent's step
    return sorted(found.values(), key=lambda s: s.order)


def order_steps(
    steps: Iterable[StepSpec],
    *,
    stages: Sequence[str],
    produced_externally: Iterable[str],
    context: str = "pipeline",
) -> list[StepSpec]:
    """Validate the DAG and return steps in execution order.

    Order is topological by frame dependencies; among ready steps, earlier
    stage first, then definition order. Raises ``DefinitionError`` for an
    unknown stage, an input nobody produces, an output produced twice, an
    output that shadows a source, a dependency that flows backwards through
    the stage order, or a cycle.
    """
    steps = list(steps)
    stage_index = {s: i for i, s in enumerate(stages)}
    external = set(produced_externally)

    producers: dict[str, StepSpec] = {}
    for s in steps:
        if s.stage not in stage_index:
            raise DefinitionError(
                f"{context}: step {s.qualified!r} declares stage {s.stage!r}; "
                f"known stages are {list(stages)}"
            )
        for out in s.outputs:
            if out in external:
                raise DefinitionError(
                    f"{context}: step {s.qualified!r} outputs {out!r}, which is a source name"
                )
            if out in producers:
                raise DefinitionError(
                    f"{context}: {out!r} is produced by both "
                    f"{producers[out].qualified!r} and {s.qualified!r}"
                )
            producers[out] = s

    for s in steps:
        for inp in s.inputs:
            if inp in external:
                continue
            producer = producers.get(inp)
            if producer is None:
                raise DefinitionError(
                    f"{context}: step {s.qualified!r} reads {inp!r}, which no source, "
                    f"step or `requires` provides"
                )
            if stage_index[producer.stage] > stage_index[s.stage]:
                raise DefinitionError(
                    f"{context}: step {s.qualified!r} ({s.stage}) reads {inp!r} produced by "
                    f"{producer.qualified!r} in the later stage {producer.stage!r}"
                )

    # Kahn's algorithm with a priority queue: (stage, definition order).
    by_id = {id(s): s for s in steps}
    indegree = {id(s): 0 for s in steps}
    dependents: dict[int, list[int]] = {id(s): [] for s in steps}
    for s in steps:
        for inp in s.inputs:
            if inp in external:
                continue
            p = producers[inp]
            indegree[id(s)] += 1
            dependents[id(p)].append(id(s))

    def key(s: StepSpec) -> tuple[int, int]:
        return (stage_index[s.stage], s.order)

    ready = [(key(s), id(s)) for s in steps if indegree[id(s)] == 0]
    heapq.heapify(ready)
    ordered: list[StepSpec] = []
    while ready:
        _, sid = heapq.heappop(ready)
        s = by_id[sid]
        ordered.append(s)
        for d in dependents[sid]:
            indegree[d] -= 1
            if indegree[d] == 0:
                heapq.heappush(ready, (key(by_id[d]), d))
    if len(ordered) != len(steps):
        stuck = [s.qualified for s in steps if indegree[id(s)] > 0]
        raise DefinitionError(f"{context}: steps form a cycle: {stuck}")
    return ordered
