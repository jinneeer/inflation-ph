"""OpenSTAT CSV ingest: exports rebuilt from the committed warehouse must round-trip exactly.

The synthetic exports mimic PXWeb downloads (title line, dotted area indentation,
'code - name' commodity labels, '..' for missing cells) in several heading layouts.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import ingest_openstat as ing  # noqa: E402

W = ing.WAREHOUSE
DIM_GEO = pd.read_parquet(W / "dim_geolocation.parquet").set_index("geolocation")["geo_level"]
DIM_COM = pd.read_parquet(W / "dim_commodity.parquet").set_index("commodity_code")["commodity"]


def _geo_label(g: str) -> str:
    return "." * int(DIM_GEO[g]) * 2 + g


def _com_label(code: str) -> str:
    return DIM_COM[code] if code == "0" else f"{code} - {DIM_COM[code]}"


def _fmt(v) -> str:
    return ".." if pd.isna(v) else repr(float(v))


def _write_wide(path: Path, df: pd.DataFrame, value: str, period, with_geo: bool = True) -> Path:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    dims = (["geolocation"] if with_geo else []) + ["commodity_code"]
    wide = df.pivot_table(index=dims, columns="date", values=value, aggfunc="first", dropna=False)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["Consumer Price Index (2018=100): synthetic export"])
        w.writerow((["Geolocation"] if with_geo else []) + ["Commodity Description"] + [period(d) for d in dates])
        for idx, row in wide.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            labels = ([_geo_label(idx[0])] if with_geo else []) + [_com_label(idx[-1])]
            w.writerow(labels + [_fmt(row[d]) for d in dates])
        w.writerow(["Source: Philippine Statistics Authority"])
    return path


def _write_two_row(path: Path, df: pd.DataFrame, value: str) -> Path:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    wide = df.pivot_table(index="commodity_code", columns="date", values=value, aggfunc="first", dropna=False)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        years = []
        for i, d in enumerate(dates):
            years.append(str(d.year) if i == 0 or dates[i - 1].year != d.year else "")
        w.writerow([""] + years)
        w.writerow(["Commodity Description"] + [d.strftime("%B") for d in dates])
        for code, row in wide.iterrows():
            w.writerow([_com_label(code)] + [_fmt(row[d]) for d in dates])
    return path


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    d = tmp_path_factory.mktemp("openstat")
    cpi = ing.read_committed("fact_cpi")
    infl = ing.read_committed("fact_inflation")
    core = ing.read_committed("fact_core")
    early = pd.to_datetime(cpi["date"]) < pd.Timestamp("2022-01-01")
    return {
        # CPI split in two downloads, months as "2018 January"
        "cpi": [_write_wide(d / "cpi_a.csv", cpi[early], "cpi", lambda t: t.strftime("%Y %B")),
                _write_wide(d / "cpi_b.csv", cpi[~early], "cpi", lambda t: t.strftime("%Y %B"))],
        "inflation": [_write_wide(d / "infl.csv", infl, "inflation_yoy_pct", lambda t: t.strftime("%YM%m"))],
        # core: national only, so no area column; months as "Jan 2018"
        "core": [_write_wide(d / "core.csv", core, "core_cpi", lambda t: t.strftime("%b %Y"), with_geo=False)],
        "core_inflation": [_write_two_row(d / "core_infl.csv",
                                          core[pd.to_datetime(core["date"]) >= ing.INFL_START],
                                          "core_inflation_yoy_pct")],
    }


def test_parse_period_layouts():
    jan = pd.Timestamp("2018-01-01")
    for s in ("2018 January", "2018 Jan", "January 2018", "Jan. 2018", "2018M01", "2018-01", "2018M1"):
        assert ing.parse_period(s) == jan, s
    for s in ("2018 Average", "2018", "Geolocation", "2018M13"):
        assert ing.parse_period(s) is None, s


def test_exports_round_trip_to_committed_warehouse(exports):
    tables, _ = ing.build(exports["cpi"], exports["core"], exports["inflation"], exports["core_inflation"])
    report = ing.compare(tables)
    for name, r in report.items():
        assert r["added"] == r["removed"] == r["changed"] == 0, (name, r)


def test_computed_rates_stay_within_rounding(exports):
    """Without the inflation exports the rates are computed from CPI: off by at most 0.1."""
    tables, notes = ing.build(exports["cpi"], exports["core"])
    assert any("computed from CPI" in n for n in notes)
    new = tables["fact_inflation"].merge(ing.read_committed("fact_inflation"),
                                         on=["date", "geolocation", "commodity_code"], suffixes=("", "_old"))
    assert len(new) == len(ing.read_committed("fact_inflation"))
    assert (new["inflation_yoy_pct"] - new["inflation_yoy_pct_old"]).abs().max() <= 0.1 + 1e-9


def test_write_produces_the_layout_the_app_reads(exports, tmp_path):
    wh = tmp_path / "warehouse"
    shutil.copytree(W, wh)
    tables, _ = ing.build(exports["cpi"], exports["core"], exports["inflation"], exports["core_inflation"], wh)
    ing.write(tables, wh)
    assert not (tmp_path / ".warehouse_staging").exists()
    assert sorted(p.name for p in (wh / "fact_cpi").iterdir()) == sorted(p.name for p in (W / "fact_cpi").iterdir())
    n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{(wh / 'fact_cpi').as_posix()}/*/*.parquet')").fetchone()[0]
    assert n == len(ing.read_committed("fact_cpi"))
    assert all(r["changed"] == r["added"] == 0 for r in ing.compare(tables, wh).values())


def test_unknown_area_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text('"Geolocation","Commodity Description","2018 January"\n'
                 '"Atlantis","0 - ALL ITEMS",100\n', encoding="utf-8")
    with pytest.raises(ing.ExportError, match="Atlantis"):
        ing.build([p], [p])
