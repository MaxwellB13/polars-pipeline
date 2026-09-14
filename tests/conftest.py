from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polspec import ColSpec, ForeignKey, FrameSpec


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


class MemReader:
    """A Reader over an in-memory frame, for tests that never touch disk."""

    def __init__(self, df: pl.DataFrame, tag: str = "mem") -> None:
        self.df = df
        self.tag = tag
        self.scans = 0

    def scan(self) -> pl.LazyFrame:
        self.scans += 1
        return self.df.lazy()

    def fingerprint(self) -> str:
        return f"{self.tag}:{self.df.hash_rows().sum()}"

    def describe(self) -> str:
        return f"mem:{self.tag}"


CUSTOMERS = pl.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Ann", "Bob", "Cy"],
        "region": ["north", "south", "east"],
    },
    schema=CustomersSpec.schema(),
)

ORDERS = pl.DataFrame(
    {
        "order_id": [10, 11, 12, 13],
        "customer_id": [1, 1, 2, 3],
        "unit_price": [9.5, 20.0, 3.25, 100.0],
        "qty": [1, 2, 10, 1],
    },
    schema=OrdersSpec.schema(),
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    CUSTOMERS.write_parquet(d / "customers.parquet")
    ORDERS.write_csv(d / "orders.csv")
    return d


@pytest.fixture
def staging_dir(tmp_path: Path) -> Path:
    return tmp_path / "staging"


@pytest.fixture
def messy_xlsx(tmp_path: Path) -> Path:
    """Two banner rows, then a header with awkward names, a blank row, and data.

    Mirrors the shape of a report exported from a BI tool.
    """
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Orders"
    ws.append(["Monthly orders report", None, None, None])
    ws.append(["Generated 2026-09-01", None, None, None])
    ws.append(["Order ID", "Customer ID", "Unit Price ", "Qty."])
    ws.append([None, None, None, None])
    ws.append([10, 1, 9.5, 1])
    ws.append([11, 1, 20.0, 2])
    ws.append([12, 2, 3.25, 10])
    ws.append([13, 3, 100.0, 1])
    ws2 = wb.create_sheet("Archive")
    ws2.append(["Archive", None, None, None])
    ws2.append(["Order ID", "Customer ID", "Unit Price ", "Qty."])
    ws2.append([1, 2, 5.0, 3])
    path = tmp_path / "orders.xlsx"
    wb.save(path)
    return path
