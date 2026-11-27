"""DuckDB connection over the Parquet warehouse. Every query in the app goes through here."""
from __future__ import annotations
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
WH = ROOT / "data" / "warehouse"
TABLES = ["dim_region", "dim_item", "dim_date", "fact_cpi", "fact_weight"]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{(WH / (t + '.parquet')).as_posix()}')")
    con.execute((ROOT / "sql" / "03_features.sql").read_text())
    scored = ROOT / "data" / "artifacts" / "nowcast_scored.parquet"
    if scored.exists():
        con.execute(f"CREATE VIEW nowcast_scored AS SELECT * FROM read_parquet('{scored.as_posix()}')")
    return con
