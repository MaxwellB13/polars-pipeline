from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pipeline import (
    BasePipeline,
    FileReader,
    HermeticError,
    PipelineConfig,
    RunOptions,
    Source,
    SourceError,
    step,
)
from tests.conftest import CUSTOMERS, ORDERS, CustomersSpec, MemReader, OrdersSpec


class Missing:
    """A reader whose file does not exist: hermetic runs must never touch it."""

    def scan(self) -> pl.LazyFrame:
        raise SourceError("hermetic run read from an external source!")

    def fingerprint(self) -> str:
        raise SourceError("hermetic run fingerprinted an external source!")

    def describe(self) -> str:
        return "missing"


class Sales(BasePipeline):
    sources = (
        Source("orders", Missing(), spec=OrdersSpec, synthetic_rows=200),
        Source("customers", Missing(), spec=CustomersSpec, synthetic_rows=20),
    )

    @step()
    def enriched(self, orders, customers):
        return orders.join(customers, on="customer_id", how="inner")

    @step()
    def totals(self, enriched):
        return enriched.group_by("region").agg((pl.col("unit_price") * pl.col("qty")).sum())


def test_hermetic_generates_fk_consistent_sources(staging_dir: Path) -> None:
    res = Sales(PipelineConfig(staging_dir)).run(hermetic=True, seed=1, mode="eager")
    pkg = res.package
    assert res.ingested == ["orders", "customers"]
    assert pkg.provenance["orders"] == "synthetic"
    orders, customers = pkg.collect("orders"), pkg.collect("customers")
    assert orders.height == 200 and customers.height == 20
    assert orders.schema == OrdersSpec.schema()
    # every order's customer exists: the inner join loses nothing
    assert pkg.materialized["enriched"].height == 200
    assert set(orders["customer_id"]) <= set(customers["customer_id"])
    assert pkg.staging is not None and pkg.staging.root == staging_dir
    assert (staging_dir / "hermetic" / "1" / "orders.parquet").exists()
    assert not (staging_dir / "orders.parquet").exists()  # real area untouched


def test_same_seed_same_data_and_reused_on_rerun(staging_dir: Path) -> None:
    p = Sales(PipelineConfig(staging_dir))
    a = p.run(hermetic=True, seed=7)
    b = p.run(hermetic=True, seed=7)
    assert b.reused == ["orders", "customers"] and b.ingested == []
    assert_frame_equal(a.package.collect("orders"), b.package.collect("orders"))

    c = p.run(hermetic=True, seed=7, refresh="all")
    assert c.ingested == ["orders", "customers"]
    assert_frame_equal(a.package.collect("orders"), c.package.collect("orders"))

    d = p.run(hermetic=True, seed=8)
    assert not a.package.collect("orders").equals(d.package.collect("orders"))


def test_unseeded_hermetic_always_regenerates(staging_dir: Path) -> None:
    p = Sales(PipelineConfig(staging_dir))
    a = p.run(hermetic=True)
    b = p.run(hermetic=True)
    assert b.ingested == ["orders", "customers"]
    assert (staging_dir / "hermetic" / "unseeded").exists()
    assert a.package.collect("orders").height == 200


def test_synthetic_rows_override(staging_dir: Path) -> None:
    res = Sales(PipelineConfig(staging_dir)).run(
        RunOptions(hermetic=True, seed=1, synthetic_rows={"orders": 5})
    )
    assert res.package.collect("orders").height == 5
    assert res.package.collect("customers").height == 20
    res = Sales(PipelineConfig(staging_dir)).run(hermetic=True, seed=2, synthetic_rows=3)
    assert res.package.collect("orders").height == 3
    assert res.package.collect("customers").height == 3


def test_reference_pins_one_real_source_inside_hermetic_run(staging_dir: Path) -> None:
    res = Sales(PipelineConfig(staging_dir)).run(
        hermetic=True, seed=3, references={"customers": CUSTOMERS}, mode="eager"
    )
    assert res.package.provenance == {
        "customers": "injected",
        "orders": "synthetic",
        "enriched": "derived",
        "totals": "derived",
    }
    assert_frame_equal(res.package.collect("customers"), CUSTOMERS)
    # generated orders only reference the three real customers
    assert set(res.package.collect("orders")["customer_id"]) <= {1, 2, 3}


def test_hermetic_requires_spec_on_every_source(staging_dir: Path) -> None:
    class NoSpec(BasePipeline):
        sources = (Source("raw", Missing()),)

        @step()
        def x(self, raw):
            return raw

    with pytest.raises(HermeticError, match=r"missing for \['raw'\]"):
        NoSpec(PipelineConfig(staging_dir)).run(hermetic=True)


def test_hermetic_then_real_share_nothing(data_dir: Path, staging_dir: Path) -> None:
    class Real(BasePipeline):
        sources = (
            Source("orders", FileReader(data_dir / "orders.csv"), spec=OrdersSpec),
            Source("customers", MemReader(CUSTOMERS), spec=CustomersSpec),
        )

        @step()
        def enriched(self, orders, customers):
            return orders.join(customers, on="customer_id")

    p = Real(PipelineConfig(staging_dir))
    synthetic = p.run(hermetic=True, seed=1)
    real = p.run()
    assert real.package.collect("orders").height == ORDERS.height
    assert synthetic.package.collect("orders").height == 1_000
    assert sorted(x.name for x in staging_dir.glob("*.parquet")) == [
        "customers.parquet",
        "orders.parquet",
    ]


def test_reference_data_is_read_for_real_in_hermetic_runs(
    data_dir: Path, staging_dir: Path
) -> None:
    """A lookup CSV in the codebase is used as-is; generated orders reference its keys."""
    customers_reader = MemReader(CUSTOMERS, "customers")

    class P(BasePipeline):
        sources = (
            Source("customers", customers_reader, spec=CustomersSpec, hermetic="real"),
            Source("orders", Missing(), spec=OrdersSpec, synthetic_rows=100),
            Source("notes", MemReader(pl.DataFrame({"k": [1]}), "notes"), hermetic="real"),
        )

        @step()
        def enriched(self, orders, customers, notes):
            return orders.join(customers, on="customer_id", how="inner")

    res = P(PipelineConfig(staging_dir)).run(hermetic=True, seed=3)
    assert customers_reader.scans == 1
    assert res.ingested == ["customers", "notes", "orders"]
    assert res.package.provenance["customers"] == "real"
    assert res.package.provenance["orders"] == "synthetic"
    assert (staging_dir / "hermetic" / "3" / "customers.parquet").exists()
    assert not (staging_dir / "customers.parquet").exists()  # real area untouched

    # FK parent was the real table: every synthetic order points at a real customer
    orders = res.package.collect("orders")
    assert set(orders["customer_id"]) <= set(CUSTOMERS["customer_id"])
    assert res.package.collect("enriched").height == 100

    # Rerun: the reference data is fingerprint-checked like any real source.
    again = P(PipelineConfig(staging_dir)).run(hermetic=True, seed=3)
    assert again.reused == ["customers", "notes", "orders"]
    assert customers_reader.scans == 1


def test_hermetic_value_validated() -> None:
    with pytest.raises(SourceError, match="hermetic must be"):
        Source("x", Missing(), hermetic="sometimes")  # type: ignore[arg-type]


def test_shared_spec_regeneration_is_not_shadowed_by_sibling(staging_dir: Path) -> None:
    """Review finding #5: a staged sibling sharing the spec must not be offered
    as a polspec reference, or it silently replaces generation."""

    class P(BasePipeline):
        sources = (
            Source("orders_a", Missing(), spec=OrdersSpec, synthetic_rows=20),
            Source("orders_b", Missing(), spec=OrdersSpec, synthetic_rows=20),
            Source("customers", Missing(), spec=CustomersSpec, synthetic_rows=5),
        )

        @step()
        def both(self, orders_a, orders_b):
            return orders_a.head(1)

    cfg = PipelineConfig(staging_dir)
    P(cfg).run(hermetic=True, seed=1)
    res = P(cfg).run(hermetic=True, seed=1, refresh={"orders_b"}, synthetic_rows={"orders_b": 7})
    assert res.ingested == ["orders_b"]
    assert res.package.collect("orders_b").height == 7
    assert res.package.collect("orders_a").height == 20


def test_registry_errors_surface_as_hermetic_errors(staging_dir: Path) -> None:
    """Review finding #6: two different specs with one class name."""
    from polspec import ColSpec, FrameSpec

    def make(dtype):
        class Spec(FrameSpec):
            x = ColSpec(dtype)

        return Spec

    class P(BasePipeline):
        sources = (
            Source("a", Missing(), spec=make(pl.Int64)),
            Source("b", Missing(), spec=make(pl.String)),
        )

        @step()
        def s(self, a, b):
            return a

    with pytest.raises(HermeticError, match="cannot build a spec registry.*both named 'Spec'"):
        P(PipelineConfig(staging_dir)).run(hermetic=True, seed=1)
