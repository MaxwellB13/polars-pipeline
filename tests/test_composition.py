from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pipeline import (
    BasePipeline,
    Child,
    DefinitionError,
    PipelineConfig,
    Source,
    step,
)
from tests.conftest import CUSTOMERS, ORDERS, CustomersSpec, MemReader, OrdersSpec


class OrdersPipeline(BasePipeline):
    """Stands alone: ingests orders, produces `clean` and `totals`."""

    sources = (Source("raw", MemReader(ORDERS, "orders"), spec=OrdersSpec),)

    @step()
    def clean(self, raw):
        return raw.filter(pl.col("qty") > 0)

    @step()
    def totals(self, clean):
        return clean.with_columns((pl.col("unit_price") * pl.col("qty")).alias("total"))


class CustomersPipeline(BasePipeline):
    requires = ("customers",)

    @step("transform", outputs="named")
    def upper(self, customers):
        return customers.with_columns(pl.col("name").str.to_uppercase())


class Sales(BasePipeline):
    sources = (Source("customers", MemReader(CUSTOMERS, "customers"), spec=CustomersSpec),)
    children = {
        "orders": OrdersPipeline,
        "cust": Child(CustomersPipeline, inputs={"customers": "customers"}),
    }

    # Dotted child names can't be read from the signature; declare them.
    @step(inputs=("orders.totals", "cust.named"))
    def enriched(self, totals, named):
        return totals.join(named, on="customer_id")


def test_flattened_plan_prefixes_child_sources_and_steps() -> None:
    plan = Sales(PipelineConfig()).plan
    assert [s.name for s in plan.sources] == ["customers", "orders.raw"]
    assert plan.requires == ()  # cust.customers is fed by the parent
    names = [b.name for b in plan.steps]
    assert names == ["orders.clean", "orders.totals", "cust.upper", "enriched"]
    by_name = {b.name: b.spec for b in plan.steps}
    assert by_name["orders.totals"].inputs == ("orders.clean",)
    assert by_name["orders.totals"].outputs == ("orders.totals",)
    assert by_name["cust.upper"].inputs == ("customers",)  # mapped to the parent's frame
    assert by_name["cust.upper"].outputs == ("cust.named",)


def test_nested_run_matches_running_by_hand(staging_dir: Path) -> None:
    res = Sales(PipelineConfig(staging_dir)).run(mode="eager")
    assert res.ingested == ["customers", "orders.raw"]
    assert res.ran == ["orders.clean", "orders.totals", "cust.upper", "enriched"]

    by_hand = (
        ORDERS.filter(pl.col("qty") > 0)
        .with_columns((pl.col("unit_price") * pl.col("qty")).alias("total"))
        .join(CUSTOMERS.with_columns(pl.col("name").str.to_uppercase()), on="customer_id")
    )
    assert_frame_equal(res.package.materialized["enriched"].sort("order_id"), by_hand)
    assert (staging_dir / "orders.raw.parquet").exists()


def test_child_still_runs_standalone(staging_dir: Path) -> None:
    res = OrdersPipeline(PipelineConfig(staging_dir)).run()
    assert res.ran == ["clean", "totals"]
    assert res.package.collect("totals").height == ORDERS.height


def test_stage_selection_and_steps_cross_nesting(staging_dir: Path) -> None:
    cfg = PipelineConfig(staging_dir)
    Sales(cfg).run(stages=["ingress"])
    res = Sales(cfg).run(steps=["orders.clean"])
    assert res.ran == ["orders.clean"]
    assert res.ingested == []


def test_hermetic_across_nesting(staging_dir: Path) -> None:
    res = Sales(PipelineConfig(staging_dir)).run(hermetic=True, seed=5, synthetic_rows=10)
    assert res.ingested == ["customers", "orders.raw"]
    assert res.package.provenance["orders.raw"] == "synthetic"
    assert res.package.collect("orders.raw").height == 10
    assert res.package.collect("enriched").height == 10


def test_child_input_mapping_to_parent_step_output(staging_dir: Path) -> None:
    class Parent(BasePipeline):
        sources = (Source("customers", MemReader(CUSTOMERS), spec=CustomersSpec),)
        children = {"c": Child(CustomersPipeline, inputs={"customers": "few"})}

        @step()
        def few(self, customers):
            return customers.head(1)

    res = Parent(PipelineConfig(staging_dir)).run()
    assert [b.name for b in Parent(PipelineConfig()).plan.steps] == ["few", "c.upper"]
    assert res.package.collect("c.named").height == 1


def test_grandchildren_flatten_recursively(staging_dir: Path) -> None:
    class Mid(BasePipeline):
        children = {"o": OrdersPipeline}

        @step(inputs=("o.totals",))
        def big(self, totals):
            return totals.filter(pl.col("total") > 10)

    class Top(BasePipeline):
        children = {"m": Mid}

        @step(inputs=("m.big",))
        def count(self, big):
            return big.select(pl.len().alias("n"))

    plan = Top(PipelineConfig()).plan
    assert [s.name for s in plan.sources] == ["m.o.raw"]
    assert [b.name for b in plan.steps] == ["m.o.clean", "m.o.totals", "m.big", "count"]
    res = Top(PipelineConfig(staging_dir)).run()
    assert res.package.collect("count").item() == 3


def test_bad_child_prefix_and_unresolved_child_input_rejected() -> None:
    with pytest.raises(DefinitionError, match="bad child prefix"):

        class P(BasePipeline):
            children = {"a.b": OrdersPipeline}

    with pytest.raises(DefinitionError, match=r"reads 'nothing'"):

        class Q(BasePipeline):
            children = {"c": Child(CustomersPipeline, inputs={"customers": "nothing"})}
