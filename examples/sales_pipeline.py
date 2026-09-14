"""End-to-end tour of polars-pipeline.

Runs one pipeline (with a child pipeline) four ways:

1. lazy, against real files (an Excel export with banner rows, and a parquet)
2. eager, same data -- every intermediate lands on ``package.materialized``
3. hermetic with seed=1, twice -- nothing external is read; results are identical
4. ``stages=["transform"]`` -- re-runs the transforms from staging without
   touching the files (proved by a counting reader)

    uv run python examples/sales_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import openpyxl
import polars as pl
from polars.testing import assert_frame_equal
from polspec import ColSpec, ForeignKey, FrameSpec

from polars_pipeline import (
    BasePipeline,
    Child,
    FileReader,
    PipelineConfig,
    RunOptions,
    Source,
    prep,
    step,
)

# --- specs: the contract every source is conformed to ------------------------


class CustomersSpec(FrameSpec):
    customer_id = ColSpec(pl.Int64, unique=True)
    name = ColSpec(pl.String)
    region = ColSpec(pl.Enum(["north", "south", "east", "west"]))


class OrdersSpec(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64)
    unit_price = ColSpec(pl.Float64, bounds=(0.0, 1000.0))
    qty = ColSpec(pl.Int64, bounds=(1, 50))

    __foreign_keys__ = [ForeignKey(["customer_id"], CustomersSpec)]


class RevenueSpec(FrameSpec):
    region = ColSpec(pl.Enum(["north", "south", "east", "west"]))
    revenue = ColSpec(pl.Float64, bounds=(0.0, None))


# --- a counting reader, to prove partial runs never touch the files -----------


class CountingFileReader(FileReader):
    scans: dict[str, int] = {}

    def scan(self) -> pl.LazyFrame:
        CountingFileReader.scans[self.path.name] = (
            CountingFileReader.scans.get(self.path.name, 0) + 1
        )
        return super().scan()


# --- pipelines ----------------------------------------------------------------


def build(data: Path) -> type[BasePipeline]:
    class Orders(BasePipeline):
        """A child pipeline: owns the messy Excel export and cleans it."""

        sources = (
            Source(
                "raw",
                CountingFileReader(data / "orders.xlsx", sheet="Orders", header_row="auto"),
                spec=OrdersSpec,
                prepare=[prep.normalise_names(), prep.drop_empty_rows()],
                synthetic_rows=500,
            ),
        )

        @step()
        def priced(self, raw):
            return raw.with_columns((pl.col("unit_price") * pl.col("qty")).alias("total"))

    class Sales(BasePipeline):
        sources = (
            Source(
                "customers",
                CountingFileReader(data / "customers.parquet"),
                spec=CustomersSpec,
                synthetic_rows=25,
            ),
        )
        children = {"orders": Child(Orders)}

        @step(inputs=("orders.priced", "customers"))
        def enriched(self, orders, customers):
            return orders.join(customers, on="customer_id", how="inner")

        @step(validate=RevenueSpec)
        def revenue(self, enriched):
            return (
                enriched.group_by("region")
                .agg(pl.col("total").sum().alias("revenue"))
                .sort("region")
            )

        @step("egress", outputs=())
        def write(self, revenue):
            revenue.sink_parquet(self.config.params["out"])

    return Sales


# --- fixture files ------------------------------------------------------------


def write_fixtures(data: Path) -> None:
    pl.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Ann", "Bob", "Cy"],
            "region": ["north", "south", "east"],
        },
        schema=CustomersSpec.schema(),
    ).write_parquet(data / "customers.parquet")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Orders"
    ws.append(["Monthly orders report"])  # banner rows a BI tool puts above the header
    ws.append(["Generated 2026-09-01"])
    ws.append(["Order ID", "Customer ID", "Unit Price ", "Qty."])
    ws.append([])
    for row in [[10, 1, 9.5, 1], [11, 1, 20.0, 2], [12, 2, 3.25, 10], [13, 3, 100.0, 1]]:
        ws.append(row)
    wb.save(data / "orders.xlsx")


# --- the tour -------------------------------------------------------------------


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # polars tables use box-drawing characters
    root = Path(tempfile.mkdtemp(prefix="polars_pipeline_"))
    data, staging, out = root / "data", root / "staging", root / "revenue.parquet"
    data.mkdir()
    write_fixtures(data)

    Sales = build(data)
    cfg = PipelineConfig(staging_dir=staging, params={"out": out})
    print(Sales(cfg).explain(), end="\n\n")

    print("1. lazy, real data")
    lazy = Sales(cfg).run(RunOptions(mode="lazy"))
    print(lazy.summary())
    print(lazy.package.collect("revenue"), end="\n\n")

    print("2. eager, real data (staging reused; intermediates materialised)")
    eager = Sales(cfg).run(mode="eager")
    print(eager.summary())
    print("materialized:", sorted(eager.package.materialized))
    assert_frame_equal(lazy.package.collect("revenue"), eager.package.materialized["revenue"])
    print()

    print("3. hermetic, seed=1, twice")
    h1 = Sales(cfg).run(hermetic=True, seed=1)
    h2 = Sales(cfg).run(hermetic=True, seed=1)
    assert_frame_equal(h1.package.collect("revenue"), h2.package.collect("revenue"))
    print(h1.summary())
    print(f"synthetic orders: {h1.package.collect('orders.raw').height} rows")
    print("second run reused:", h2.reused)
    print(h1.package.collect("revenue"), end="\n\n")

    print("4. transform stage only, from staging")
    before = dict(CountingFileReader.scans)
    partial = Sales(cfg).run(stages=["transform"])
    assert CountingFileReader.scans == before, "partial run read a file!"
    print(partial.summary())
    print("file reads during partial run:", 0, end="\n\n")

    print(f"staged files under {staging}:")
    for p in sorted(staging.rglob("*.parquet")):
        print("  ", p.relative_to(staging))


if __name__ == "__main__":
    main()
