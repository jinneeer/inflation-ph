"""ETL: PSA OpenSTAT CSV exports -> clean star schema (Parquet) via DuckDB SQL.

    python -m src.etl            # reads data/raw/*.csv, writes data/warehouse/*.parquet

Raw exports are semicolon-separated, 2 title lines, wide (one column per month).
Files are detected by CONTENT (title line + first month column), not file name,
because exports are easy to mislabel.
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, WH = ROOT / "data" / "raw", ROOT / "data" / "warehouse"
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def read_wide(path: Path) -> tuple[str, pd.DataFrame]:
    title = path.read_text(encoding="utf-8-sig").splitlines()[0].strip().strip('"')
    df = pd.read_csv(path, sep=";", skiprows=2, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    geo = df.columns[0]
    df[geo] = df[geo].str.strip().str.replace(r"^\.+", "", regex=True).replace({"PHILIPPINES": "Philippines"})
    return title, df


def kind(title: str, df: pd.DataFrame) -> str:
    if title.startswith("Weights"):
        return "weights"
    return "yoy" if title.startswith("Year-on-Year") else "cpi"


def melt(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(id_vars=list(df.columns[:2]), var_name="ym", value_name="value")
    long.columns = ["geo_raw", "item_raw", "ym", "value"]
    p = long["ym"].str.extract(r"(\d{4})\s+(\w{3})")
    long["date"] = pd.to_datetime(dict(year=p[0].astype(int), month=p[1].map(MONTHS), day=1))
    return long.drop(columns="ym")


def load_raw() -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {"cpi": [], "yoy": [], "weights": []}
    for f in sorted(RAW.glob("*.csv")):
        title, df = read_wide(f)
        k = kind(title, df)
        out[k].append(df if k == "weights" else melt(df))
        print(f"  {f.name:34s} -> {k:8s} {df.shape}")
    cpi = pd.concat(out["cpi"]).drop_duplicates(["geo_raw", "item_raw", "date"])
    yoy = pd.concat(out["yoy"]).drop_duplicates(["geo_raw", "item_raw", "date"])
    return {"cpi": cpi, "yoy": yoy, "weights": out["weights"][0]}


def main() -> None:
    WH.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    con = duckdb.connect()
    for k, v in raw.items():
        con.register(f"raw_{k}", v)
    con.execute((ROOT / "sql" / "01_build_warehouse.sql").read_text())
    for t in ["dim_region", "dim_item", "dim_date", "fact_cpi", "fact_weight"]:
        con.execute(f"COPY {t} TO '{WH / (t + '.parquet')}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t:12s} {n:>7,} rows")
    # data-quality gates
    chk = con.execute((ROOT / "sql" / "02_quality_checks.sql").read_text()).fetchall()
    bad = [r for r in chk if r[1] != 0]
    for name, n in chk:
        print(f"  check {name:34s} {'OK' if n == 0 else 'FAIL ' + str(n)}")
    if bad:
        sys.exit("data-quality checks failed")


if __name__ == "__main__":
    main()
