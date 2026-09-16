from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polars_pipeline import (
    BasePipeline,
    DefinitionError,
    FunctionReader,
    PipelineConfig,
    Source,
    SourceError,
    step,
)
from tests.conftest import CUSTOMERS, ORDERS, CustomersSpec, OrdersSpec


class FakeDB:
    """Counts queries; `version` stands in for the data changing upstream."""

    def __init__(self) -> None:
        self.queries = 0
        self.version = 1

    def orders(self) -> pl.DataFrame:
        self.queries += 1
        return ORDERS

    def watermark(self) -> int:
        return self.version


def test_function_reader_wraps_call_and_conforms(staging_dir: Path) -> None:
    db = FakeDB()
    src = Source("orders", FunctionReader(db.orders, key={"sql": "orders"}), spec=OrdersSpec)
    assert src.read().collect().schema == OrdersSpec.schema()
    assert db.queries == 1
    assert "call:" in src.reader.describe()


def test_function_reader_rejects_non_frames() -> None:
    reader = FunctionReader(lambda: [1, 2, 3], name="bad")
    with pytest.raises(SourceError, match="expected a polars frame"):
        reader.scan()


def test_fingerprint_tracks_key_and_watermark() -> None:
    db = FakeDB()
    a = FunctionReader(db.orders, key={"sql": "a"}, name="x").fingerprint()
    b = FunctionReader(db.orders, key={"sql": "b"}, name="x").fingerprint()
    assert a != b
    with_mark = FunctionReader(db.orders, key={"sql": "a"}, name="x", watermark=db.watermark)
    m1 = with_mark.fingerprint()
    db.version = 2
    assert with_mark.fingerprint() != m1
    assert db.queries == 0  # fingerprinting never runs the main query


def test_sources_as_method_sees_config(staging_dir: Path) -> None:
    db = FakeDB()

    class P(BasePipeline):
        def sources(self):
            year = self.config.params["year"]
            return (
                Source(
                    "orders",
                    FunctionReader(db.orders, key={"year": year}, watermark=db.watermark),
                    spec=OrdersSpec,
                ),
                Source("customers", FunctionReader(lambda: CUSTOMERS), spec=CustomersSpec),
            )

        @step()
        def joined(self, orders, customers):
            return orders.join(customers, on="customer_id")

    cfg = PipelineConfig(staging_dir, params={"year": 2026})
    res = P(cfg).run()
    assert res.ingested == ["orders", "customers"] and db.queries == 1

    # Same key, same watermark: staging is reused, the query is not re-run.
    assert P(cfg).run().reused == ["orders", "customers"] and db.queries == 1

    # Upstream data moved: only orders is re-queried.
    db.version = 2
    res = P(cfg).run()
    assert res.ingested == ["orders"] and db.queries == 2

    # Different partition: the key changed, so it re-queries.
    P(PipelineConfig(staging_dir, params={"year": 2025})).run()
    assert db.queries == 3

    # Hermetic: no query at all.
    before = db.queries
    res = P(cfg).run(hermetic=True, seed=1)
    assert db.queries == before
    assert res.package.collect("joined").height > 0


def test_sources_method_still_validated_at_instantiation(staging_dir: Path) -> None:
    class P(BasePipeline):
        def sources(self):
            return (
                Source("a", FunctionReader(lambda: ORDERS)),
                Source("a", FunctionReader(lambda: ORDERS)),
            )

    with pytest.raises(DefinitionError, match="duplicate source names"):
        P(PipelineConfig(staging_dir))

    class Q(BasePipeline):
        def sources(self):
            return (Source("a", FunctionReader(lambda: ORDERS)),)

        @step()
        def x(self, nope):
            return nope

    with pytest.raises(DefinitionError, match="reads 'nope'"):
        Q(PipelineConfig(staging_dir))
