from __future__ import annotations

import pytest

from polars_pipeline import BasePipeline, DefinitionError, PipelineConfig, Source, step
from tests.conftest import CUSTOMERS, CustomersSpec, MemReader, OrdersSpec


def test_steps_ordered_topologically_with_definition_tiebreak() -> None:
    class P(BasePipeline):
        sources = (Source("a", MemReader(CUSTOMERS)),)

        @step(outputs="z")
        def last(self, y):
            return y

        @step()
        def first(self, a):
            return a

        @step(outputs="y")
        def middle(self, first):
            return first

        @step(outputs="w")
        def independent(self, a):
            return a

    names = [b.name for b in P(PipelineConfig()).plan.steps]
    # `last` is defined first, so once `middle` unblocks it, it beats `independent`.
    assert names == ["first", "middle", "last", "independent"]


def test_inputs_inferred_from_signature_and_outputs_default_to_name() -> None:
    class P(BasePipeline):
        sources = (
            Source("orders", MemReader(CUSTOMERS)),
            Source("customers", MemReader(CUSTOMERS)),
        )

        @step()
        def enriched(self, orders, customers, threshold=3):
            return orders

    (s,) = P._steps
    assert s.inputs == ("orders", "customers")
    assert s.outputs == ("enriched",)
    assert s.stage == "transform"


def test_unknown_input_is_definition_error() -> None:
    with pytest.raises(DefinitionError, match=r"reads 'nope', which no source"):

        class P(BasePipeline):
            sources = (Source("a", MemReader(CUSTOMERS)),)

            @step()
            def x(self, nope):
                return nope


def test_requires_declares_external_inputs() -> None:
    class P(BasePipeline):
        requires = ("supplied",)

        @step()
        def x(self, supplied):
            return supplied

    assert P(PipelineConfig()).plan.requires == ("supplied",)


def test_duplicate_output_and_source_shadowing_rejected() -> None:
    with pytest.raises(DefinitionError, match="produced by both"):

        class P(BasePipeline):
            sources = (Source("a", MemReader(CUSTOMERS)),)

            @step(outputs="o")
            def x(self, a):
                return a

            @step(outputs="o")
            def y(self, a):
                return a

    with pytest.raises(DefinitionError, match="which is a source name"):

        class Q(BasePipeline):
            sources = (Source("a", MemReader(CUSTOMERS)),)

            @step(outputs="a")
            def x(self, a):
                return a


def test_cycle_rejected() -> None:
    with pytest.raises(DefinitionError, match="cycle"):

        class P(BasePipeline):
            @step(outputs="b")
            def x(self, c):
                return c

            @step(outputs="c")
            def y(self, b):
                return b


def test_backward_stage_dependency_rejected() -> None:
    with pytest.raises(DefinitionError, match="later stage 'egress'"):

        class P(BasePipeline):
            sources = (Source("a", MemReader(CUSTOMERS)),)

            @step("egress", outputs="e")
            def out(self, a):
                return a

            @step("transform")
            def t(self, e):
                return e


def test_unknown_stage_and_missing_ingress_rejected() -> None:
    with pytest.raises(DefinitionError, match="declares stage 'nope'"):

        class P(BasePipeline):
            sources = (Source("a", MemReader(CUSTOMERS)),)

            @step("nope")
            def x(self, a):
                return a

    with pytest.raises(DefinitionError, match="must include 'ingress'"):

        class Q(BasePipeline):
            stages = ("transform",)


def test_validate_requires_single_output() -> None:
    with pytest.raises(DefinitionError, match="exactly one output"):

        @step(outputs=("a", "b"), validate=OrdersSpec)
        def f(self, x):
            return x, x


def test_subclass_overrides_parent_step() -> None:
    class Base(BasePipeline):
        sources = (Source("a", MemReader(CUSTOMERS), spec=CustomersSpec),)

        @step()
        def clean(self, a):
            return a

    class Derived(Base):
        @step()
        def clean(self, a):
            return a.head(1)

    assert [s.name for s in Derived._steps] == ["clean"]
    assert Derived._steps[0].fn is Derived.__dict__["clean"]


def test_explain_lists_sources_and_steps() -> None:
    class P(BasePipeline):
        name = "demo"
        sources = (Source("a", MemReader(CUSTOMERS), spec=CustomersSpec),)

        @step()
        def x(self, a):
            return a

    text = P(PipelineConfig()).explain()
    assert "a: mem:mem spec=CustomersSpec" in text
    assert "[transform] x: a -> x" in text
