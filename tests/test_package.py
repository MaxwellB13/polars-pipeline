from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pipeline import FileReader, PipelineConfig, PipelineError, Source
from polars_pipeline.package import DataPackage
from tests.conftest import CUSTOMERS, ORDERS, CustomersSpec


def test_from_frames_accepts_eager_and_lazy(tmp_path: Path) -> None:
    pkg = DataPackage.from_frames(
        PipelineConfig(tmp_path), {"customers": CUSTOMERS, "orders": ORDERS.lazy()}
    )
    assert isinstance(pkg["customers"], pl.LazyFrame)
    assert isinstance(pkg["orders"], pl.LazyFrame)
    assert "customers" in pkg.materialized  # eager input is kept as-is
    assert "orders" not in pkg.materialized
    assert pkg.provenance == {"customers": "supplied", "orders": "supplied"}
    assert_frame_equal(pkg.collect("orders"), ORDERS)
    assert "orders" in pkg.materialized


def test_from_sources_wires_staging(data_dir: Path, staging_dir: Path) -> None:
    src = Source("customers", FileReader(data_dir / "customers.parquet"), spec=CustomersSpec)
    pkg = DataPackage.from_sources(PipelineConfig(staging_dir), [src])
    assert pkg.staging is not None and pkg.staging.root == staging_dir
    assert pkg.names() == []
    with pytest.raises(PipelineError, match="duplicate source"):
        DataPackage.from_sources(PipelineConfig(staging_dir), [src, src])


def test_get_unknown_raises_with_hint(tmp_path: Path) -> None:
    pkg = DataPackage.from_frames(PipelineConfig(tmp_path), {"a": CUSTOMERS})
    with pytest.raises(PipelineError, match=r"no frame named 'b'.*has: a"):
        pkg.get("b")


def test_put_replaces_and_invalidates_cache(tmp_path: Path) -> None:
    pkg = DataPackage.from_frames(PipelineConfig(tmp_path), {"a": CUSTOMERS})
    pkg.put("a", CUSTOMERS.lazy().head(1))
    assert "a" not in pkg.materialized
    assert pkg.collect("a").height == 1
    assert pkg.provenance["a"] == "derived"
    with pytest.raises(PipelineError, match="must be a polars"):
        pkg.put("x", [1, 2, 3])  # type: ignore[arg-type]


def test_collect_all_batches(tmp_path: Path) -> None:
    pkg = DataPackage.from_frames(
        PipelineConfig(tmp_path), {"c": CUSTOMERS.lazy(), "o": ORDERS.lazy()}
    )
    out = pkg.collect_all()
    assert set(out) == {"c", "o"}
    assert_frame_equal(out["c"], CUSTOMERS)
