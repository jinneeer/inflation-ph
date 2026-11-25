"""Rebuild the model-ready regional feature table from the Parquet warehouse.

Reads data/warehouse/fact_inflation + dim_geolocation, computes lags, rolling
statistics and the next-month target with DuckDB window functions, and writes
data/warehouse/feat_region_month.parquet.

    python -m pipeline.build_features            # rebuild (overwrites)
    python -m pipeline.build_features --check    # rebuild in memory, diff against the committed file

Raw PSA extracts (the OpenSTAT PXWeb downloads) are not part of this repository;
the pipeline therefore starts from the committed warehouse (see README, "Reproducibility").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "data" / "warehouse"
OUT = WAREHOUSE / "feat_region_month.parquet"
FIRST_MONTH = "2019-01-01"   # YoY inflation exists from 2019 (2018 is the base year)

FEATURE_SQL = f"""
WITH base AS (
  SELECT f.date, f.geolocation, f.inflation_yoy_pct AS infl
  FROM read_parquet('{WAREHOUSE.as_posix()}/fact_inflation/*/*.parquet', hive_partitioning = true) f
  JOIN read_parquet('{WAREHOUSE.as_posix()}/dim_geolocation.parquet') g USING (geolocation)
  WHERE f.commodity_code = '0'            -- all-items index
    AND g.geo_type = 'region'             -- the 18 administrative regions
    AND f.date >= CAST('{FIRST_MONTH}' AS DATE)
    AND f.inflation_yoy_pct IS NOT NULL
)
SELECT date, geolocation, infl,
  lag(infl, 1)  OVER w AS lag1,
  lag(infl, 2)  OVER w AS lag2,
  lag(infl, 3)  OVER w AS lag3,
  lag(infl, 6)  OVER w AS lag6,
  lag(infl, 12) OVER w AS lag12,
  avg(infl) OVER (PARTITION BY geolocation ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)  AS ma3,
  avg(infl) OVER (PARTITION BY geolocation ORDER BY date ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)  AS ma6,
  avg(infl) OVER (PARTITION BY geolocation ORDER BY date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS ma12,
  stddev_samp(infl) OVER (PARTITION BY geolocation ORDER BY date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS vol12,
  month(date) AS month_num,
  sin(2 * pi() * month(date) / 12) AS month_sin,
  cos(2 * pi() * month(date) / 12) AS month_cos,
  year(date) - 2019 AS time_idx,
  lead(infl, 1) OVER w AS target_next
FROM base
WINDOW w AS (PARTITION BY geolocation ORDER BY date)
ORDER BY geolocation, date
"""


def build(con: duckdb.DuckDBPyConnection | None = None):
    con = con or duckdb.connect()
    return con.execute(FEATURE_SQL).df()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed parquet instead of writing")
    args = ap.parse_args()
    df = build()
    print(f"built {len(df):,} rows, {df['geolocation'].nunique()} regions, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    if args.check:
        import pandas as pd
        old = pd.read_parquet(OUT).sort_values(["geolocation", "date"]).reset_index(drop=True)
        new = df.sort_values(["geolocation", "date"]).reset_index(drop=True)
        new["date"] = pd.to_datetime(new["date"])
        old["date"] = pd.to_datetime(old["date"])
        same_shape = old.shape == new.shape and list(old.columns) == list(new.columns)
        diff = float((old.drop(columns=["geolocation", "date"]) - new.drop(columns=["geolocation", "date"])).abs().max().max()) if same_shape else float("nan")
        print(f"shape match: {same_shape}; max abs difference vs committed file: {diff:.3g}")
        return 0 if same_shape and diff < 1e-9 else 1
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
