# polars-pipeline

A small framework for writing Polars pipelines that all look the same, so any of them can be:

- run **hermetically** — no database, SharePoint or file reads; every source is replaced by data generated from its [polspec](https://pypi.org/project/polspec/) spec,
- run **lazily** in production or **eagerly** when debugging, with identical step code,
- fed from a **staging area** — every source lands once as Parquet and is scanned lazily from there,
- **composed** from other pipelines,
- run **partially** (`stages=`, `steps=`) and **selectively re-ingested** (`refresh=`).

```
Source ──(read → prepare → conform)──▶ StagingArea (Parquet + manifest) ──scan──▶ DataPackage.frames
                                                                                       │
                                                       @step methods, run as a DAG ────┘──▶ outputs
```

## Install

```bash
uv add polars-pipeline            # add [excel] for .xlsx sources
```

## Writing a pipeline

```python
import polars as pl
from polspec import ColSpec, ForeignKey, FrameSpec
from polars_pipeline import BasePipeline, FileReader, PipelineConfig, Source, prep, step


class CustomersSpec(FrameSpec):
    customer_id = ColSpec(pl.Int64, unique=True)
    region = ColSpec(pl.Enum(["north", "south"]))


class OrdersSpec(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64)
    total = ColSpec(pl.Float64, bounds=(0.0, None))
    __foreign_keys__ = [ForeignKey(["customer_id"], CustomersSpec)]


class Sales(BasePipeline):
    sources = (
        Source("customers", FileReader(r"\\nas\share\customers.parquet"), spec=CustomersSpec),
        Source(
            "orders",
            FileReader(r"\\nas\share\orders.xlsx", sheet="Orders", header_row="auto"),
            spec=OrdersSpec,
            prepare=[prep.normalise_names(), prep.drop_empty_rows()],
        ),
    )

    @step()                                   # inputs come from the parameter names
    def enriched(self, orders, customers):
        return orders.join(customers, on="customer_id")

    @step(outputs=("by_region", "by_customer"))
    def summaries(self, enriched):
        return (
            enriched.group_by("region").agg(pl.col("total").sum()),
            enriched.group_by("customer_id").agg(pl.col("total").sum()),
        )

    @step("egress", outputs=())
    def write(self, by_region):
        by_region.sink_parquet(self.config.params["out"])


pipeline = Sales(PipelineConfig(staging_dir=".staging", params={"out": "revenue.parquet"}))
```

Steps always receive and return `LazyFrame`s. Return one frame, a tuple in `outputs` order, a dict keyed by output name, or `None` for a step with no outputs. Add `validate=SomeSpec` to check a step's output against a polspec spec.

### Ingress: read → prepare → conform → stage

Raw files are rarely stage-ready. Each source goes through a fixed path:

1. **read** — `FileReader` scans parquet/csv/ipc/ndjson lazily and reads Excel eagerly. For Excel, `sheet=` takes a name, index, or list of names (concatenated with a `__sheet` column) and `header_row="auto"` drops banner rows above the real header.
2. **prepare** — any `LazyFrame -> LazyFrame` callables. `polars_pipeline.prep` ships `normalise_names`, `drop_empty_rows`, `drop_empty_columns`, `promote_header`, `rename`, `select_columns`, `cast`, `strip_strings`, `filter_rows`.
3. **conform** — select the spec's columns in spec order and cast to its dtypes (`conform="strict"` fails on bad values, `"lenient"` nulls them, `"none"` skips). The staged Parquet therefore *always* has the spec's exact schema — the contract every step and every hermetic run relies on.
4. **stage** — written to `<staging_dir>/<name>.parquet` and recorded in `manifest.json` with the reader's fingerprint and a hash of the prepare configuration.

New kinds of source (database, SharePoint, HTTP) implement the three-method `Reader` protocol: `scan()`, `fingerprint()`, `describe()`.

## Running

```python
pipeline.run()                                    # lazy, ingest what's stale, run everything
pipeline.run(mode="eager")                        # collect after every step; see package.materialized
pipeline.run(hermetic=True, seed=1)               # nothing external is read
pipeline.run(stages=["transform"])                # reuse staging, skip ingress and egress
pipeline.run(steps=["enriched"])                  # one step, from staged inputs
pipeline.run(refresh="all")                       # force re-ingest of every source
pipeline.run(refresh={"orders"})                  # ... or just one
pipeline.run(references={"customers": my_df})     # inject a frame in place of a source
pipeline.run(package=DataPackage.from_frames(cfg, {"orders": df, "customers": df2}))
```

`run()` returns a `RunResult`: `.package` (the frames), `.ingested` / `.reused` (what ingress did), `.ran` / `.skipped` / `.failed`, and `.summary()`.

**Lazy vs eager.** Lazy mode hands each step Parquet scans and never collects; the caller collects what it wants from `result.package`. Eager mode collects every step's output as it is produced, so an error is raised by the step that caused it (`StepError`, with the original exception chained) and every intermediate is on `package.materialized`.

**Refresh.** `"stale"` (default) re-ingests a source when its file changed or its `prepare`/`conform` configuration changed; `"none"` only ingests what is missing; `"all"` or a set of names forces it.

**Hermetic.** Every source needs a `spec`. Specs are generated together through a `polspec.Registry`, parents before children, so foreign keys hold by construction. Synthetic data is staged under `<staging_dir>/hermetic/<seed>/` — never over real data — and a seeded run is reused until `refresh` says otherwise. `synthetic_rows=` (int or per-source mapping) overrides each source's default. `references=` pins a real or hand-built frame for some sources while the rest are generated.

## Composition

```python
class Reporting(BasePipeline):
    sources = (Source("customers", ..., spec=CustomersSpec),)
    children = {
        "sales": Sales,                                             # own sources, own steps
        "churn": Child(ChurnPipeline, inputs={"customers": "customers"}),  # fed by the parent
    }

    @step(inputs=("sales.by_region", "churn.rate"))
    def dashboard(self, by_region, rate):
        ...
```

A child's sources and steps are flattened into the parent under a dotted prefix (`sales.orders`, `sales.enriched`). One flat DAG means stage selection, refresh, hermetic generation and eager debugging behave identically at any depth, and the child still runs standalone. A child declaring `requires = ("customers",)` expects the parent (or a `from_frames` package) to supply that frame.

## Example

```bash
uv run python examples/sales_pipeline.py
```

Runs one pipeline lazily, eagerly, hermetically (twice, asserting equality), and transform-only from staging with a counting reader proving no file was opened.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
```
