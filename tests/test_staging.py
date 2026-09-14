from __future__ import annotations

import json
import os
import time
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pipeline import FileReader, Source, StagingError, prep
from polars_pipeline.staging import StagingArea
from tests.conftest import CUSTOMERS, ORDERS, CustomersSpec, OrdersSpec


def _orders(data_dir: Path, **kw) -> Source:
    return Source("orders", FileReader(data_dir / "orders.csv"), spec=OrdersSpec, **kw)


def test_stage_scan_roundtrip(data_dir: Path, staging_dir: Path) -> None:
    area = StagingArea(staging_dir)
    src = _orders(data_dir)
    entry = area.stage(
        src, src.read(), fingerprint=src.reader.fingerprint(), prepare_hash=src.prepare_hash()
    )
    assert entry.rows == 4
    assert entry.provenance == "real"
    assert area.has("orders")
    assert_frame_equal(area.scan("orders").collect(), ORDERS)
    assert (staging_dir / "orders.parquet").exists()
    assert not [p for p in staging_dir.iterdir() if p.name.startswith(".orders")]


def test_manifest_survives_reopen(data_dir: Path, staging_dir: Path) -> None:
    src = _orders(data_dir)
    StagingArea(staging_dir).stage(src, src.read(), fingerprint="fp", prepare_hash="ph")
    again = StagingArea(staging_dir)
    assert again.names() == ["orders"]
    e = again.entry("orders")
    assert (e.fingerprint, e.prepare_hash) == ("fp", "ph")
    assert e.schema["unit_price"] == "Float64"
    raw = json.loads((staging_dir / "manifest.json").read_text())
    assert raw["version"] == 1


def test_scan_unstaged_raises(staging_dir: Path) -> None:
    with pytest.raises(StagingError, match="not staged"):
        StagingArea(staging_dir).scan("nope")


def test_is_stale_detects_file_and_prepare_changes(data_dir: Path, staging_dir: Path) -> None:
    area = StagingArea(staging_dir)
    src = _orders(data_dir)
    assert area.is_stale(src)  # nothing staged yet
    area.stage(
        src, src.read(), fingerprint=src.reader.fingerprint(), prepare_hash=src.prepare_hash()
    )
    assert not area.is_stale(src)

    changed_prepare = _orders(data_dir, prepare=prep.strip_strings())
    assert area.is_stale(changed_prepare)

    time.sleep(0.01)
    path = data_dir / "orders.csv"
    with path.open("a") as fh:
        fh.write("14,3,1.0,1\n")
    os.utime(path, None)
    assert area.is_stale(src)


def test_synthetic_entries_never_stale(data_dir: Path, staging_dir: Path) -> None:
    area = StagingArea(staging_dir).hermetic(seed=7)
    src = Source("customers", FileReader(data_dir / "customers.parquet"), spec=CustomersSpec)
    area.stage(src, CUSTOMERS, provenance="synthetic", seed=7)
    assert area.root == staging_dir / "hermetic" / "7"
    assert not area.is_stale(src)
    assert area.entry("customers").reader is None


def test_drop_removes_file_and_entry(data_dir: Path, staging_dir: Path) -> None:
    area = StagingArea(staging_dir)
    src = _orders(data_dir)
    area.stage(src, src.read())
    area.drop("orders")
    assert not area.has("orders")
    assert not (staging_dir / "orders.parquet").exists()
    area.drop("orders")  # idempotent


def test_failed_write_leaves_nothing_behind(data_dir: Path, staging_dir: Path) -> None:
    area = StagingArea(staging_dir)
    src = _orders(data_dir)
    bad = pl.LazyFrame({"x": ["1", "a"]}).select(pl.col("x").cast(pl.Int64))
    with pytest.raises(StagingError):
        area.stage(src, bad)
    assert not area.has("orders")
    assert list(staging_dir.glob("*.parquet")) == []
