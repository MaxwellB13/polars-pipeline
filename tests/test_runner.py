from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polspec import ColSpec, FrameSpec

from polars_pipeline import (
    BasePipeline,
    DataPackage,
    FileReader,
    MissingInputError,
    PipelineConfig,
    RunOptions,
    Source,
    StagingError,
    StepError,
    prep,
    step,
)
from tests.conftest import CUSTOMERS, ORDERS, CustomersSpec, MemReader, OrdersSpec


class EnrichedSpec(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64)
    name = ColSpec(pl.String)
    total = ColSpec(pl.Float64, bounds=(0.0, None))


def make_pipeline(orders_reader, customers_reader, **source_kw):
    class Sales(BasePipeline):
        name = "sales"
        sources = (
            Source("orders", orders_reader, spec=OrdersSpec, **source_kw),
            Source("customers", customers_reader, spec=CustomersSpec),
        )

        @step(validate=EnrichedSpec)
        def enriched(self, orders, customers):
            return orders.join(customers, on="customer_id").select(
                "order_id",
                "customer_id",
                "name",
                (pl.col("unit_price") * pl.col("qty")).alias("total"),
            )

        @step(outputs=("by_customer", "by_region"))
        def summaries(self, enriched, customers):
            by_customer = enriched.group_by("customer_id").agg(pl.col("total").sum())
            by_region = (
                enriched.join(customers.select("customer_id", "region"), on="customer_id")
                .group_by("region")
                .agg(pl.col("total").sum())
            )
            return by_customer, by_region

        @step("egress", outputs=())
        def write(self, by_customer):
            out = self.config.params.get("out")
            if out:
                by_customer.sink_parquet(out)

    return Sales


EXPECTED_ENRICHED = pl.DataFrame(
    {
        "order_id": [10, 11, 12, 13],
        "customer_id": [1, 1, 2, 3],
        "name": ["Ann", "Ann", "Bob", "Cy"],
        "total": [9.5, 40.0, 32.5, 100.0],
    }
)


@pytest.fixture
def sales(data_dir: Path):
    return make_pipeline(
        FileReader(data_dir / "orders.csv"), FileReader(data_dir / "customers.parquet")
    )


def test_lazy_and_eager_agree(sales, staging_dir: Path, tmp_path: Path) -> None:
    cfg = PipelineConfig(staging_dir, params={"out": tmp_path / "out.parquet"})
    lazy = sales(cfg).run(RunOptions(mode="lazy"))
    eager = sales(cfg).run(RunOptions(mode="eager"))

    assert lazy.ingested == ["orders", "customers"]
    assert eager.reused == ["orders", "customers"]  # second run reuses staging
    assert lazy.ran == ["enriched", "summaries", "write"]
    assert lazy.package.materialized == {}
    assert set(eager.package.materialized) >= {"enriched", "by_customer", "by_region"}

    a = lazy.package.collect("enriched").sort("order_id")
    b = eager.package.materialized["enriched"].sort("order_id")
    assert_frame_equal(a, b)
    assert_frame_equal(a, EXPECTED_ENRICHED)
    assert (tmp_path / "out.parquet").exists()
    assert "[transform] enriched: ran" in eager.summary()


def test_run_accepts_option_kwargs(sales, staging_dir: Path) -> None:
    res = sales(PipelineConfig(staging_dir)).run(mode="eager", stages=["ingress"])
    assert res.ran == []
    assert res.ingested == ["orders", "customers"]


def test_transform_only_reuses_staging_without_reading(staging_dir: Path) -> None:
    orders_reader = MemReader(ORDERS, "orders")
    customers_reader = MemReader(CUSTOMERS, "customers")
    P = make_pipeline(orders_reader, customers_reader)
    cfg = PipelineConfig(staging_dir)

    P(cfg).run(stages=["ingress"])
    assert (orders_reader.scans, customers_reader.scans) == (1, 1)

    res = P(cfg).run(stages=["transform"])
    assert (orders_reader.scans, customers_reader.scans) == (1, 1)
    assert res.ran == ["enriched", "summaries"]
    assert res.skipped == ["write"]
    assert sorted(res.reused) == ["customers", "orders"]
    assert_frame_equal(res.package.collect("enriched").sort("order_id"), EXPECTED_ENRICHED)


def test_steps_selection_runs_named_steps_only(sales, staging_dir: Path) -> None:
    cfg = PipelineConfig(staging_dir)
    sales(cfg).run()
    res = sales(cfg).run(steps=["enriched"])
    assert res.ran == ["enriched"]
    assert res.ingested == []


def test_missing_input_error_names_the_producer(sales, staging_dir: Path) -> None:
    with pytest.raises(MissingInputError, match=r"needs 'orders'.*run the 'ingress' stage"):
        sales(PipelineConfig(staging_dir)).run(stages=["transform"])
    sales(PipelineConfig(staging_dir)).run(stages=["ingress"])
    with pytest.raises(MissingInputError, match=r"needs 'enriched'.*produced by step 'enriched'"):
        sales(PipelineConfig(staging_dir)).run(steps=["summaries"])


def test_unknown_stage_step_or_reference_rejected(sales, staging_dir: Path) -> None:
    p = sales(PipelineConfig(staging_dir))
    with pytest.raises(MissingInputError, match="unknown stage"):
        p.run(stages=["nope"])
    with pytest.raises(MissingInputError, match="unknown step"):
        p.run(steps=["nope"])
    with pytest.raises(MissingInputError, match="unknown sources"):
        p.run(references={"nope": CUSTOMERS})


def test_refresh_policies(data_dir: Path, staging_dir: Path) -> None:
    orders_reader = MemReader(ORDERS, "orders")
    customers_reader = MemReader(CUSTOMERS, "customers")
    P = make_pipeline(orders_reader, customers_reader)
    cfg = PipelineConfig(staging_dir)

    P(cfg).run()
    P(cfg).run(refresh="none")
    assert (orders_reader.scans, customers_reader.scans) == (1, 1)

    P(cfg).run(refresh={"orders"})
    assert (orders_reader.scans, customers_reader.scans) == (2, 1)

    res = P(cfg).run(refresh="all")
    assert (orders_reader.scans, customers_reader.scans) == (3, 2)
    assert res.ingested == ["orders", "customers"]

    # "stale": only the source whose data changed is re-read.
    orders_reader.df = ORDERS.head(2)
    res = P(cfg).run(refresh="stale")
    assert (orders_reader.scans, customers_reader.scans) == (4, 2)
    assert res.ingested == ["orders"] and res.reused == ["customers"]
    assert res.package.collect("orders").height == 2


def test_prepare_change_triggers_restage(staging_dir: Path) -> None:
    orders_reader = MemReader(ORDERS, "orders")
    cfg = PipelineConfig(staging_dir)
    make_pipeline(orders_reader, MemReader(CUSTOMERS))(cfg).run()
    assert orders_reader.scans == 1
    make_pipeline(orders_reader, MemReader(CUSTOMERS), prepare=prep.strip_strings())(cfg).run()
    assert orders_reader.scans == 2


def test_file_change_triggers_restage(sales, data_dir: Path, staging_dir: Path) -> None:
    cfg = PipelineConfig(staging_dir)
    sales(cfg).run()
    time.sleep(0.01)
    path = data_dir / "orders.csv"
    with path.open("a") as fh:
        fh.write("14,3,1.0,1\n")
    os.utime(path, None)
    res = sales(cfg).run()
    assert res.ingested == ["orders"]
    assert res.package.collect("orders").height == 5


def test_reference_frame_replaces_source_and_is_conformed(sales, staging_dir: Path) -> None:
    cfg = PipelineConfig(staging_dir)
    one = ORDERS.head(1)
    res = sales(cfg).run(references={"orders": one})
    assert res.package.provenance["orders"] == "injected"
    assert res.package.collect("enriched").height == 1
    assert res.package.staging.entry("orders").provenance == "injected"

    bad = ORDERS.with_columns(pl.col("unit_price").cast(pl.String).str.replace("9.5", "x"))
    with pytest.raises(StagingError, match="conversion from `str` to `f64` failed"):
        sales(cfg).run(references={"orders": bad}, refresh="all")


def test_from_frames_package_skips_reading(staging_dir: Path) -> None:
    orders_reader = MemReader(ORDERS, "orders")
    P = make_pipeline(orders_reader, MemReader(CUSTOMERS))
    cfg = PipelineConfig(staging_dir)
    pkg = DataPackage.from_frames(cfg, {"orders": ORDERS, "customers": CUSTOMERS.lazy()})
    res = P(cfg).run(package=pkg, mode="eager")
    assert orders_reader.scans == 0
    assert res.package.provenance["orders"] == "injected"
    assert_frame_equal(res.package.materialized["enriched"].sort("order_id"), EXPECTED_ENRICHED)


def test_step_error_wraps_and_reports(staging_dir: Path) -> None:
    class Broken(BasePipeline):
        sources = (Source("orders", MemReader(ORDERS), spec=OrdersSpec),)

        @step()
        def boom(self, orders):
            return orders.select("no_such_column")

        @step()
        def after(self, boom):
            return boom

    p = Broken(PipelineConfig(staging_dir))
    # Lazy: the plan builds; the error only surfaces on collect.
    res = p.run(mode="lazy")
    assert res.ran == ["boom", "after"]
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        res.package.collect("after")
    # Eager: the failing step is named and later steps never run.
    with pytest.raises(StepError, match="step 'boom' failed") as exc:
        p.run(mode="eager")
    assert isinstance(exc.value.__cause__, pl.exceptions.ColumnNotFoundError)


def test_validate_hook_fails_bad_output_in_eager_mode(staging_dir: Path) -> None:
    class Negative(BasePipeline):
        sources = (
            Source("orders", MemReader(ORDERS), spec=OrdersSpec),
            Source("customers", MemReader(CUSTOMERS), spec=CustomersSpec),
        )

        @step(validate=EnrichedSpec)
        def enriched(self, orders, customers):
            return orders.join(customers, on="customer_id").select(
                "order_id", "customer_id", "name", pl.lit(-1.0).alias("total")
            )

    with pytest.raises(StepError, match="enriched"):
        Negative(PipelineConfig(staging_dir)).run(mode="eager")


def test_dict_outputs_and_bad_return_shapes(staging_dir: Path) -> None:
    class P(BasePipeline):
        sources = (Source("orders", MemReader(ORDERS), spec=OrdersSpec),)

        @step(outputs=("a", "b"))
        def split(self, orders):
            return {"b": orders.tail(1), "a": orders.head(1)}

        @step(outputs="c")
        def wrong(self, orders):
            return 42

    res = P(PipelineConfig(staging_dir)).run(steps=["split"], stages=["ingress", "transform"])
    assert res.package.collect("a").height == 1
    with pytest.raises(StepError, match="must return a LazyFrame"):
        P(PipelineConfig(staging_dir)).run(steps=["wrong"])


def test_result_summary_and_terminal_outputs(sales, staging_dir: Path) -> None:
    p = sales(PipelineConfig(staging_dir))
    res = p.run()
    assert p.plan.terminal_outputs() == ("by_region",)
    text = res.summary()
    assert "mode=lazy" in text and "[ingress] orders: ran" in text
