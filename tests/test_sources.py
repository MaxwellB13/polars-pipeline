from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pipeline import FileReader, Source, SourceError, prep
from tests.conftest import CUSTOMERS, ORDERS, CustomersSpec, OrdersSpec


def test_parquet_and_csv_scan_to_expected_schema(data_dir: Path) -> None:
    customers = FileReader(data_dir / "customers.parquet").scan().collect()
    assert customers.schema == CustomersSpec.schema()
    orders = FileReader(data_dir / "orders.csv").scan().collect()
    assert orders.columns == ORDERS.columns


def test_format_inferred_or_required(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="cannot infer format"):
        FileReader(tmp_path / "thing.dat")
    r = FileReader(tmp_path / "thing.dat", format="csv")
    assert r.format == "csv"


def test_fingerprint_changes_when_file_changes(data_dir: Path) -> None:
    path = data_dir / "orders.csv"
    r = FileReader(path)
    before = r.fingerprint()
    assert r.fingerprint() == before
    time.sleep(0.01)
    with path.open("a") as fh:
        fh.write("14,3,1.0,1\n")
    os.utime(path, None)
    assert r.fingerprint() != before


def test_fingerprint_includes_read_options(data_dir: Path) -> None:
    path = data_dir / "orders.csv"
    a = FileReader(path).fingerprint()
    b = FileReader(path, read_kwargs={"skip_rows": 1}).fingerprint()
    assert a != b


def test_source_conforms_csv_to_spec(data_dir: Path) -> None:
    src = Source("orders", FileReader(data_dir / "orders.csv"), spec=OrdersSpec)
    out = src.read().collect()
    assert out.schema == OrdersSpec.schema()
    assert_frame_equal(out, ORDERS)


def test_conform_reports_missing_columns(data_dir: Path) -> None:
    src = Source(
        "orders",
        FileReader(data_dir / "orders.csv"),
        spec=OrdersSpec,
        prepare=prep.rename({"qty": "quantity"}),
    )
    with pytest.raises(SourceError, match=r"missing after prepare: \['qty'\]"):
        src.read().collect_schema()


def test_strict_conform_raises_lenient_nulls(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("order_id,customer_id,unit_price,qty\n1,1,abc,1\n")
    strict = Source("orders", FileReader(path), spec=OrdersSpec)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        strict.read().collect()
    lenient = Source("orders", FileReader(path), spec=OrdersSpec, conform="lenient")
    out = lenient.read().collect()
    assert out["unit_price"].to_list() == [None]


def test_messy_excel_becomes_spec_shaped(messy_xlsx: Path) -> None:
    src = Source(
        "orders",
        FileReader(messy_xlsx, sheet="Orders", header_row="auto"),
        spec=OrdersSpec,
        prepare=[prep.normalise_names(), prep.drop_empty_rows()],
    )
    out = src.read().collect()
    assert out.schema == OrdersSpec.schema()
    assert_frame_equal(out, ORDERS)


def test_multi_sheet_excel_tags_sheet(messy_xlsx: Path) -> None:
    reader = FileReader(messy_xlsx, sheet=["Orders", "Archive"], header_row="auto")
    out = prep.drop_empty_rows(subset=("Order ID",))(reader.scan()).collect()
    assert "__sheet" in out.columns
    assert out["__sheet"].to_list() == ["Orders"] * 4 + ["Archive"]


def test_normalise_names_dedupes_and_snake_cases() -> None:
    lf = pl.LazyFrame(
        {"Unit Price ": [1], "unit_price": [2], "Qty.": [3], "1st": [4], "__sheet": ["a"]}
    )
    out = prep.normalise_names()(lf).collect()
    assert out.columns == ["unit_price", "unit_price_2", "qty", "_1st", "__sheet"]


def test_prepare_hash_tracks_prepare_config(data_dir: Path) -> None:
    reader = FileReader(data_dir / "customers.parquet")
    a = Source("c", reader, spec=CustomersSpec).prepare_hash()
    b = Source("c", reader, spec=CustomersSpec, prepare=prep.strip_strings()).prepare_hash()
    c = Source("c", reader, spec=CustomersSpec, conform="lenient").prepare_hash()
    assert len({a, b, c}) == 3

    def f(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    d = Source("c", reader, spec=CustomersSpec, prepare=f).prepare_hash()
    assert d == Source("c", reader, spec=CustomersSpec, prepare=f).prepare_hash()
    assert d != a


def test_reader_protocol_enforced(data_dir: Path) -> None:
    class NotAReader:
        def scan(self) -> pl.LazyFrame:
            return CUSTOMERS.lazy()

    with pytest.raises(SourceError, match="reader must implement"):
        Source("c", NotAReader())  # type: ignore[arg-type]


def test_promote_header_dedupes_repeated_labels() -> None:
    """Review finding #8."""
    lf = pl.LazyFrame(
        {
            "column_1": ["report", "Q1", "1"],
            "column_2": [None, "Total", "2"],
            "column_3": ["x", "Q2", "3"],
            "column_4": [None, "Total", "4"],
        }
    )
    out = prep.promote_header()(lf).collect()
    assert out.columns == ["Q1", "Total", "Q2", "Total_2"]
    assert out.height == 1
