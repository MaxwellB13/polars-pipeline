"""A pipeline over several data sources, some of which need cleaning before staging.

Sources
-------
headcount   Excel export from an HR system: two banner rows above the header,
            column names with spaces and units, one tab per region.
cost_centres CSV on a network drive: fine as-is, but codes need trimming.
timesheets  Parquet: clean, but large, so it should stay lazy.

Flow
----
ingress:    each source is read -> prepared -> conformed to its spec -> staged as Parquet
transform:  join the three, derive utilisation, drop rows that fail business rules
egress:     write the result

    uv run python examples/workforce_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import polars as pl
from polspec import ColSpec, ForeignKey, FrameSpec

from polars_pipeline import BasePipeline, FileReader, PipelineConfig, Source, prep, step

# ---------------------------------------------------------------------------
# 1. Specs: what each source must look like *after* cleaning.
#    The staged Parquet always has exactly this schema, and hermetic runs
#    generate data from it.
# ---------------------------------------------------------------------------


class CostCentreSpec(FrameSpec):
    cost_centre = ColSpec(pl.String, unique=True)
    department = ColSpec(pl.String)
    budget_hours = ColSpec(pl.Int64, bounds=(0, 100_000))


class HeadcountSpec(FrameSpec):
    employee_id = ColSpec(pl.Int64, unique=True)
    cost_centre = ColSpec(pl.String)
    fte = ColSpec(pl.Float64, bounds=(0.0, 1.0))
    region = ColSpec(pl.Enum(["emea", "apac"]))

    __foreign_keys__ = [ForeignKey(["cost_centre"], CostCentreSpec)]


class TimesheetSpec(FrameSpec):
    employee_id = ColSpec(pl.Int64)
    week = ColSpec(pl.Int64, bounds=(1, 52))
    hours = ColSpec(pl.Float64, bounds=(0.0, 80.0))

    __foreign_keys__ = [ForeignKey(["employee_id"], HeadcountSpec)]


class UtilisationSpec(FrameSpec):
    """Contract for the pipeline's output; checked by the step that produces it."""

    cost_centre = ColSpec(pl.String)
    department = ColSpec(pl.String)
    fte = ColSpec(pl.Float64, bounds=(0.0, None))
    hours = ColSpec(pl.Float64, bounds=(0.0, None))
    utilisation = ColSpec(pl.Float64, bounds=(0.0, None))


# ---------------------------------------------------------------------------
# 2. Source-specific cleanup that's too bespoke for the shipped prep helpers.
#    Any LazyFrame -> LazyFrame callable works here.
# ---------------------------------------------------------------------------


def fte_from_percent(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The HR export gives FTE as '80%' strings and region as the tab name."""
    return lf.with_columns(
        pl.col("fte_percent").str.strip_suffix("%").cast(pl.Float64).truediv(100).alias("fte"),
        pl.col("__sheet").str.to_lowercase().alias("region"),
    )


# ---------------------------------------------------------------------------
# 3. The pipeline. `build` takes the paths so the same class works against the
#    network drive in production and a temp dir here.
# ---------------------------------------------------------------------------


def build(root: Path) -> type[BasePipeline]:
    class Workforce(BasePipeline):
        name = "workforce"

        sources = (
            Source(
                "cost_centres",
                FileReader(root / "cost_centres.csv"),
                spec=CostCentreSpec,
                prepare=[prep.strip_strings()],  # ' CC-100 ' -> 'CC-100'
                synthetic_rows=8,
            ),
            Source(
                "headcount",
                FileReader(
                    root / "headcount.xlsx",
                    sheet=["EMEA", "APAC"],  # one tab per region, concatenated
                    header_row="auto",  # skip the banner rows above the header
                ),
                spec=HeadcountSpec,
                prepare=[
                    prep.normalise_names(),  # 'Employee ID' -> employee_id, 'FTE (%)' -> fte
                    prep.rename({"fte": "fte_percent"}),
                    prep.drop_empty_rows(subset=("employee_id",)),
                    prep.strip_strings(),
                    fte_from_percent,
                ],
                synthetic_rows=40,
            ),
            Source(
                "timesheets",
                FileReader(root / "timesheets.parquet"),
                spec=TimesheetSpec,
                synthetic_rows=400,
            ),
        )

        # -- transform -------------------------------------------------------
        # Inputs are frame names in the package: sources, or earlier outputs.
        # Every step gets LazyFrames and returns LazyFrames.

        @step()
        def hours_by_employee(self, timesheets):
            weeks = self.config.params.get("weeks")
            if weeks:
                timesheets = timesheets.filter(pl.col("week").is_in(weeks))
            return timesheets.group_by("employee_id").agg(pl.col("hours").sum())

        @step()
        def staffed(self, headcount, cost_centres, hours_by_employee):
            return (
                headcount.join(cost_centres, on="cost_centre", how="left")
                .join(hours_by_employee, on="employee_id", how="left")
                .with_columns(pl.col("hours").fill_null(0.0))
            )

        @step(outputs=("utilisation", "unmapped"))
        def split(self, staffed):
            """Rows whose cost centre isn't in the master go to a side output."""
            unmapped = staffed.filter(pl.col("department").is_null())
            mapped = staffed.filter(pl.col("department").is_not_null())
            utilisation = (
                mapped.group_by("cost_centre", "department")
                .agg(pl.col("fte").sum(), pl.col("hours").sum())
                .with_columns((pl.col("hours") / (pl.col("fte") * 37.5 * 4)).alias("utilisation"))
                .sort("cost_centre")
            )
            return utilisation, unmapped

        @step(validate=UtilisationSpec)
        def checked(self, utilisation):
            # A pure contract check: validate= runs polspec against the output.
            return utilisation.select(UtilisationSpec.schema().names())

        # -- egress ----------------------------------------------------------

        @step("egress", outputs=())
        def write(self, checked, unmapped):
            out = Path(self.config.params["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            checked.sink_parquet(out / "utilisation.parquet")
            unmapped.sink_parquet(out / "unmapped.parquet")

    return Workforce


# ---------------------------------------------------------------------------
# Fixture files standing in for the network drive.
# ---------------------------------------------------------------------------


def write_fixtures(root: Path) -> None:
    import openpyxl

    (root / "cost_centres.csv").write_text(
        "cost_centre,department,budget_hours\n CC-100 ,Engineering,6000\nCC-200,Finance,1500\n"
    )

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    rows = {
        "EMEA": [[1, "CC-100", "100%"], [2, "CC-100", "80%"], [3, "CC-999", "50%"]],
        "APAC": [[4, "CC-200", "100%"]],
    }
    for tab, data in rows.items():
        ws = wb.create_sheet(tab)
        ws.append(["Headcount extract"])  # banner
        ws.append([f"Region: {tab}"])  # banner
        ws.append(["Employee ID", "Cost Centre", "FTE (%)"])
        ws.append([])  # blank row under the header
        for r in data:
            ws.append(r)
    wb.save(root / "headcount.xlsx")

    pl.DataFrame(
        {
            "employee_id": [1, 1, 2, 3, 4, 4],
            "week": [1, 2, 1, 1, 1, 2],
            "hours": [37.5, 40.0, 30.0, 10.0, 37.5, 37.5],
        }
    ).write_parquet(root / "timesheets.parquet")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(tempfile.mkdtemp(prefix="workforce_"))
    write_fixtures(root)
    Workforce = build(root)
    cfg = PipelineConfig(
        staging_dir=root / "staging", params={"out_dir": root / "out", "weeks": [1, 2]}
    )
    pipeline = Workforce(cfg)

    print(pipeline.explain(), "\n")

    # Debugging: eager, so any failure names its step and intermediates are inspectable.
    res = pipeline.run(mode="eager")
    print(res.summary(), "\n")
    print("staged headcount (cleaned + conformed):")
    print(res.package.collect("headcount"), "\n")
    print("utilisation:")
    print(res.package.materialized["checked"], "\n")
    print("unmapped cost centres:")
    print(res.package.materialized["unmapped"].select("employee_id", "cost_centre"), "\n")

    # Production: lazy; staging is reused because nothing changed.
    res = pipeline.run()
    print("lazy rerun reused:", res.reused)

    # Hermetic: no files read; FK-consistent synthetic data from the specs.
    res = pipeline.run(hermetic=True, seed=42)
    print("hermetic utilisation rows:", res.package.collect("checked").height)


if __name__ == "__main__":
    main()
