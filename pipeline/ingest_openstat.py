"""Rebuild the Parquet warehouse from CSV files exported by hand from PSA OpenSTAT.

OpenSTAT (https://openstat.psa.gov.ph) is a PXWeb site: pick a table, select the
variables, then download it as "Comma delimited with heading" (CSV). Export:

    --cpi        CPI (2018=100) by geolocation and commodity group   (required)
    --core       Core CPI (2018=100) by commodity group              (required)
    --inflation  Inflation rates (2018=100) by geolocation/commodity (optional:
                 computed from --cpi, rounded to 1 decimal, when left out)
    --core-inflation  Core inflation rates                           (optional:
                 computed from --core the same way when left out)

Each option can be given more than once, because OpenSTAT limits how many cells one
download may hold (split a big table by years or by area and pass every file).

    python -m pipeline.ingest_openstat --cpi exports/cpi.csv --core exports/core.csv \
        --inflation exports/inflation.csv --check     # compare with the committed warehouse
    python -m pipeline.ingest_openstat --cpi ... --core ... --inflation ...   # write

What it writes in data/warehouse: fact_cpi/, fact_inflation/, fact_core/ (one
year=YYYY/part0.parquet per year), dim_date.parquet and fact_cpi_history.parquet.
dim_geolocation and dim_commodity (2018 basket weights) are reference tables and are
kept as they are; any area or commodity in an export that they do not list is an error.
fact_cpi_history keeps its committed CPI before 2018 (an older PSA series linked to
2018=100), takes 2018 onward from the new national all-items CPI, and recomputes the
year-on-year rate from that CPI.

Scope matches the committed warehouse: every commodity for PHILIPPINES, all items only
for regions and provinces. Rows outside it are dropped and counted.

After writing: python -m pipeline.build_features, python -m models.train_models, and set
data_vintage in app/config.py to the date of the export.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import io
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "data" / "warehouse"
NATIONAL = "PHILIPPINES"
ALL_ITEMS = "0"
BASE_START = pd.Timestamp("2018-01-01")      # first month of the 2018=100 series
INFL_START = pd.Timestamp("2019-01-01")      # first year-on-year rate on that base
MISSING = {"", "..", "...", "-", "--", "na", "n/a", "nan", "…"}

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9


class ExportError(ValueError):
    """An export that cannot be read or does not match the reference tables."""


# --------------------------------------------------------------------------- parsing

def parse_period(cell: str) -> pd.Timestamp | None:
    """'2018 January', '2018 Jan', 'January 2018', '2018M01', '2018-01' -> first of month."""
    s = str(cell).strip().strip('"').strip()
    m = re.fullmatch(r"(\d{4})\s*(?:M|-|/|\s)\s*(\d{1,2})", s, re.I)
    if m:
        y, mo = int(m[1]), int(m[2])
    else:
        m = re.fullmatch(r"(\d{4})\s+([A-Za-z]+)\.?", s) or re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{4})", s)
        if not m:
            return None
        a, b = m[1], m[2]
        y, name = (int(a), b) if a.isdigit() else (int(b), a)
        mo = MONTHS.get(name.lower())
        if mo is None:
            return None              # e.g. "2018 Average": annual columns are ignored
    if not (1 <= mo <= 12 and 1900 <= y <= 2100):
        return None
    return pd.Timestamp(year=y, month=mo, day=1)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def _rows(text: str) -> list[list[str]]:
    sample = text[:20000]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delim = ","
    return [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), delimiter=delim)]


def _value(cell: str) -> float:
    s = str(cell).strip().replace(" ", "")
    if s.lower() in MISSING:
        return np.nan
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return np.nan


def _dim_roles(header: list[str]) -> tuple[int | None, int | None]:
    """Which of the left-hand columns hold the area and the commodity."""
    geo = com = None
    for i, h in enumerate(header):
        k = h.lower()
        if geo is None and ("geo" in k or "area" in k or "region" in k or "province" in k):
            geo = i
        elif com is None and ("commodity" in k or "item" in k or "description" in k or "group" in k):
            com = i
    rest = [i for i in range(len(header)) if i not in (geo, com)]
    if com is None and rest:
        com = rest.pop()             # the innermost (last) dimension is the commodity
    if geo is None and rest:
        geo = rest.pop()
    return geo, com


def read_openstat(path: Path) -> pd.DataFrame:
    """One OpenSTAT CSV export -> long rows [geolocation, commodity_label, date, value]."""
    rows = _rows(_decode(Path(path).read_bytes()))
    for r, row in enumerate(rows):
        low = [c.lower() for c in row]
        # long layout: a header with Year and Period/Month columns
        if "year" in low and any(k in low for k in ("period", "month")):
            return _read_long(path, rows[r], rows[r + 1:])
        periods = [parse_period(c) for c in row]
        if any(p is not None for p in periods):
            first = next(i for i, p in enumerate(periods) if p is not None)
            return _read_wide(path, row[:first], periods, rows[r + 1:], first)
        # two header rows: years (forward-filled) above month names
        if r + 1 < len(rows) and any(re.fullmatch(r"\d{4}", c) for c in row):
            nxt, year, combined = rows[r + 1], None, []
            for i in range(max(len(row), len(nxt))):
                y = row[i] if i < len(row) else ""
                year = y if re.fullmatch(r"\d{4}", y) else year
                mo = nxt[i] if i < len(nxt) else ""
                combined.append(parse_period(f"{year} {mo}") if year and mo else None)
            if any(p is not None for p in combined):
                first = next(i for i, p in enumerate(combined) if p is not None)
                header = [nxt[i] or row[i] for i in range(first)]
                return _read_wide(path, header, combined, rows[r + 2:], first)
    raise ExportError(f"{path}: no month columns found (expected headings like '2018 January')")


def _read_wide(path, header, periods, body, first) -> pd.DataFrame:
    gi, ci = _dim_roles(header)
    if ci is None:
        raise ExportError(f"{path}: no commodity column left of the month columns")
    out, last = [], [""] * first
    for row in body:
        if not any(row):
            continue
        dims = [(row[i] if i < len(row) else "") or last[i] for i in range(first)]  # blank = same as above
        last = dims
        if len(row) <= first:
            continue                 # footnote or source line
        for i in range(first, len(row)):
            p = periods[i] if i < len(periods) else None
            if p is not None:
                out.append((dims[gi] if gi is not None else NATIONAL, dims[ci], p, _value(row[i])))
    return pd.DataFrame(out, columns=["geolocation", "commodity_label", "date", "value"])


def _read_long(path, header, body) -> pd.DataFrame:
    low = [h.lower() for h in header]
    yi = low.index("year")
    pi = low.index("period") if "period" in low else low.index("month")
    vi = len(header) - 1
    dims = [i for i in range(len(header)) if i not in (yi, pi, vi)]
    g, c = _dim_roles([header[i] for i in dims])
    gi = dims[g] if g is not None else None
    ci = dims[c] if c is not None else None
    if ci is None:
        raise ExportError(f"{path}: no commodity column")
    out = []
    for row in body:
        if len(row) <= max(yi, pi, vi):
            continue
        p = parse_period(f"{row[yi]} {row[pi]}")
        if p is not None:
            out.append((row[gi] if gi is not None else NATIONAL, row[ci], p, _value(row[vi])))
    return pd.DataFrame(out, columns=["geolocation", "commodity_label", "date", "value"])


# --------------------------------------------------------------------------- mapping

def _norm(s: str) -> str:
    s = re.sub(r"^[\s. ]+", "", str(s))          # OpenSTAT indents areas with dots
    return re.sub(r"\s+", " ", s).strip().casefold()


def map_geolocation(df: pd.DataFrame, dim_geo: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    lookup = {_norm(g): g for g in dim_geo["geolocation"]}
    lookup.setdefault("philippines", NATIONAL)
    df["geolocation"] = df["geolocation"].map(lambda g: lookup.get(_norm(g)))
    return df, sorted(set(df.loc[df["geolocation"].isna(), "_raw_geo"]))


def map_commodity(labels: pd.Series, dim_com: pd.DataFrame):
    codes = set(dim_com["commodity_code"])
    by_name: dict[str, list[str]] = {}
    for code, name in zip(dim_com["commodity_code"], dim_com["commodity"]):
        by_name.setdefault(_norm(name), []).append(code)
    result, unknown, ambiguous = {}, [], []
    for label in labels.unique():
        s = re.sub(r"^[\s. ]+", "", str(label))
        m = re.match(r"(\d{1,2}(?:\.\d+)*)\s*(?:[-–:]\s*|\s+|$)(.*)$", s)
        code = None
        if m:
            raw = m[1]
            head, *tail = raw.split(".")
            for cand in (raw, ".".join([head.zfill(2), *tail]), ".".join([head.lstrip("0") or "0", *tail])):
                if cand in codes:
                    code = cand
                    break
            name = m[2]
        else:
            name = s
        if code is None:
            hits = by_name.get(_norm(name), [])
            if len(hits) == 1:
                code = hits[0]
            elif hits:
                ambiguous.append(label)
            else:
                unknown.append(label)
        result[label] = code
    return labels.map(result), unknown, ambiguous


def load_exports(paths: list[Path], dim_geo: pd.DataFrame, dim_com: pd.DataFrame,
                 value_col: str, notes: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = read_openstat(p)
        if df.empty:
            raise ExportError(f"{p}: no data rows")
        df["_raw_geo"] = df["geolocation"]
        df, bad_geo = map_geolocation(df, dim_geo)
        if bad_geo:
            raise ExportError(f"{p}: areas not in dim_geolocation: {bad_geo[:10]}"
                              f"{' …' if len(bad_geo) > 10 else ''}")
        df["commodity_code"], unknown, ambiguous = map_commodity(df["commodity_label"], dim_com)
        if unknown or ambiguous:
            raise ExportError(f"{p}: commodities not in dim_commodity: {unknown[:10]}; "
                              f"names matching more than one code (export with codes): {ambiguous[:10]}")
        notes.append(f"{Path(p).name}: {len(df):,} cells, {df['date'].min():%Y-%m} → {df['date'].max():%Y-%m}")
        frames.append(df[["date", "geolocation", "commodity_code", "value"]])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    # the same cell in two overlapping exports must agree
    dup = df.duplicated(["date", "geolocation", "commodity_code"], keep=False)
    if dup.any():
        spread = df[dup].groupby(["date", "geolocation", "commodity_code"])["value"].agg(lambda v: v.max() - v.min())
        if (spread > 1e-9).any():
            k = spread[spread > 1e-9].index[0]
            raise ExportError(f"overlapping exports disagree on {value_col} for {k}")
        df = df.drop_duplicates(["date", "geolocation", "commodity_code"])
    return df.rename(columns={"value": value_col})


# --------------------------------------------------------------------------- building

def _in_scope(df: pd.DataFrame, name: str, notes: list[str]) -> pd.DataFrame:
    keep = (df["geolocation"] == NATIONAL) | (df["commodity_code"] == ALL_ITEMS)
    keep &= df["date"] >= BASE_START
    if (~keep).sum():
        notes.append(f"{name}: dropped {(~keep).sum():,} cells outside the warehouse scope")
    return df[keep]


def _yoy(df: pd.DataFrame, keys: list[str], col: str, decimals: int | None = 1) -> pd.Series:
    prev = df[keys + ["date", col]].copy()
    prev["date"] = prev["date"] + pd.DateOffset(years=1)
    merged = df[keys + ["date", col]].merge(prev, on=keys + ["date"], how="left", suffixes=("", "_prev"))
    rate = (merged[col] / merged[col + "_prev"] - 1) * 100
    return (rate.round(decimals) if decimals is not None else rate).to_numpy()


def build(cpi: list[Path], core: list[Path], inflation: list[Path] = (), core_inflation: list[Path] = (),
          warehouse: Path = WAREHOUSE) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read the exports and return the warehouse tables (not written) and notes."""
    notes: list[str] = []
    dim_geo = pd.read_parquet(warehouse / "dim_geolocation.parquet")
    dim_com = pd.read_parquet(warehouse / "dim_commodity.parquet")
    key = ["date", "geolocation", "commodity_code"]

    fact_cpi = _in_scope(load_exports(cpi, dim_geo, dim_com, "cpi", notes), "cpi", notes)
    if (fact_cpi["cpi"] <= 0).any():
        raise ExportError("CPI export holds zero or negative index values: is it the CPI table?")

    if inflation:
        fact_inf = _in_scope(load_exports(inflation, dim_geo, dim_com, "inflation_yoy_pct", notes), "inflation", notes)
        fact_inf = fact_inf[fact_inf["date"] >= INFL_START]
        calc = fact_cpi.assign(calc=_yoy(fact_cpi, ["geolocation", "commodity_code"], "cpi", None))
        check = fact_inf.merge(calc, on=key)
        off = (check["inflation_yoy_pct"] - check["calc"]).abs() > 0.15
        if off.mean() > 0.01:
            raise ExportError(f"{off.sum():,} of {len(check):,} inflation rates differ from the CPI by more "
                              "than rounding: were the CPI and inflation tables exported for the same base?")
    else:
        fact_inf = fact_cpi.assign(inflation_yoy_pct=_yoy(fact_cpi, ["geolocation", "commodity_code"], "cpi"))
        fact_inf = fact_inf[fact_inf["date"] >= INFL_START].dropna(subset=["inflation_yoy_pct"])
        notes.append("inflation: computed from CPI (rounded to 1 decimal); PSA's published rate can differ by 0.1")

    core_df = load_exports(core, dim_geo, dim_com, "core_cpi", notes)
    core_df = core_df[(core_df["geolocation"] == NATIONAL) & (core_df["date"] >= BASE_START)]
    if core_inflation:
        ci = load_exports(core_inflation, dim_geo, dim_com, "core_inflation_yoy_pct", notes)
        ci = ci[(ci["geolocation"] == NATIONAL) & (ci["date"] >= INFL_START)]
        core_df = core_df.merge(ci, on=key, how="left")
    else:
        core_df["core_inflation_yoy_pct"] = _yoy(core_df, ["commodity_code"], "core_cpi")
        core_df.loc[core_df["date"] < INFL_START, "core_inflation_yoy_pct"] = np.nan
        notes.append("core inflation: computed from core CPI (rounded to 1 decimal)")
    fact_core = core_df[["date", "commodity_code", "core_cpi", "core_inflation_yoy_pct"]]

    nat = fact_cpi[(fact_cpi["geolocation"] == NATIONAL) & (fact_cpi["commodity_code"] == ALL_ITEMS)]
    if nat.empty:
        raise ExportError(f"the CPI export has no {NATIONAL} / ALL ITEMS row")
    months = pd.date_range(BASE_START, nat["date"].max(), freq="MS")
    gaps = sorted(set(months) - set(nat["date"]))
    if gaps:
        raise ExportError(f"national all-items CPI is missing months: {[f'{g:%Y-%m}' for g in gaps[:12]]}")

    dim_date = pd.DataFrame({"date": months})
    dim_date["year"] = dim_date["date"].dt.year.astype("int64")
    dim_date["month"] = dim_date["date"].dt.month.astype("int64")
    dim_date["quarter"] = dim_date["date"].dt.quarter.astype("int64")

    old_hist = pd.read_parquet(warehouse / "fact_cpi_history.parquet")
    old_hist["date"] = pd.to_datetime(old_hist["date"])
    hist = pd.concat([old_hist.loc[old_hist["date"] < BASE_START, ["date", "cpi"]],
                      nat[["date", "cpi"]]], ignore_index=True).sort_values("date")
    # the history's rate is computed from its own CPI (unrounded), not PSA's published rate
    hist["inflation_yoy_pct"] = _yoy(hist.assign(k=0), ["k"], "cpi", None)
    hist["year"] = hist["date"].dt.year.astype("int64")
    fact_cpi_history = hist[["date", "year", "cpi", "inflation_yoy_pct"]].reset_index(drop=True)

    tables = {
        "fact_cpi": fact_cpi[key + ["cpi"]],
        "fact_inflation": fact_inf[key + ["inflation_yoy_pct"]],
        "fact_core": fact_core,
        "dim_date": dim_date,
        "fact_cpi_history": fact_cpi_history,
    }
    for name, df in tables.items():
        df = df.sort_values([c for c in ("date", "geolocation", "commodity_code") if c in df]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        tables[name] = df
    return tables, notes


PARTITIONED = {"fact_cpi", "fact_inflation", "fact_core"}


def read_committed(name: str, warehouse: Path = WAREHOUSE) -> pd.DataFrame:
    if name in PARTITIONED:
        df = pd.concat([pd.read_parquet(p) for p in sorted((warehouse / name).glob("year=*/*.parquet"))],
                       ignore_index=True)
    else:
        df = pd.read_parquet(warehouse / f"{name}.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def compare(tables: dict[str, pd.DataFrame], warehouse: Path = WAREHOUSE) -> dict[str, dict]:
    """Rows added, removed and changed (beyond 1e-9) per table, against the committed files."""
    report = {}
    for name, new in tables.items():
        old = read_committed(name, warehouse)
        keys = [c for c in ("date", "geolocation", "commodity_code") if c in new]
        vals = [c for c in new.columns if c not in keys and c != "year"] if name != "dim_date" else ["year", "month", "quarter"]
        m = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
        both = m[m["_merge"] == "both"]
        changed = np.zeros(len(both), dtype=bool)
        for v in vals:
            a, b = both[v + "_old"].astype(float), both[v + "_new"].astype(float)
            changed |= ((a - b).abs() > 1e-9).to_numpy() | (a.isna() != b.isna()).to_numpy()
        report[name] = {"rows": len(new), "added": int((m["_merge"] == "right_only").sum()),
                        "removed": int((m["_merge"] == "left_only").sum()), "changed": int(changed.sum())}
    return report


def write(tables: dict[str, pd.DataFrame], warehouse: Path = WAREHOUSE) -> None:
    """Write every table to a staging folder first, then swap each one into place."""
    staging = warehouse.parent / f".{warehouse.name}_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name, df in tables.items():
        if name in PARTITIONED:
            for year, part in df.groupby(pd.to_datetime(df["date"]).dt.year):
                d = staging / name / f"year={year}"
                d.mkdir(parents=True)
                part.to_parquet(d / "part0.parquet", index=False)
        else:
            df.to_parquet(staging / f"{name}.parquet", index=False)
    for name in tables:
        src = staging / (name if name in PARTITIONED else f"{name}.parquet")
        dst = warehouse / src.name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        src.rename(dst)
    shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpi", action="append", type=Path, required=True, help="CPI export (repeatable)")
    ap.add_argument("--core", action="append", type=Path, required=True, help="core CPI export (repeatable)")
    ap.add_argument("--inflation", action="append", type=Path, default=[], help="inflation export (repeatable)")
    ap.add_argument("--core-inflation", action="append", type=Path, default=[], help="core inflation export")
    ap.add_argument("--warehouse", type=Path, default=WAREHOUSE, help="warehouse folder (default data/warehouse)")
    ap.add_argument("--check", action="store_true", help="compare with the committed warehouse, write nothing")
    a = ap.parse_args(argv)
    try:
        tables, notes = build(a.cpi, a.core, a.inflation, a.core_inflation, a.warehouse)
    except (ExportError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for n in notes:
        print(f"  {n}")
    latest = max(tables["fact_cpi"]["date"])
    print(f"latest month: {latest:%Y-%m}")
    report = compare(tables, a.warehouse)
    print(f"{'table':<18}{'rows':>8}{'added':>8}{'removed':>9}{'changed':>9}")
    for name, r in report.items():
        print(f"{name:<18}{r['rows']:>8,}{r['added']:>8,}{r['removed']:>9,}{r['changed']:>9,}")
    if a.check:
        same = all(r["added"] == r["removed"] == r["changed"] == 0 for r in report.values())
        print("identical to the committed warehouse" if same else "differs from the committed warehouse (nothing written)")
        return 0 if same else 1
    write(tables, a.warehouse)
    print(f"wrote {len(tables)} tables to {a.warehouse}")
    print("next: python -m pipeline.build_features && python -m models.train_models, "
          f"then set data_vintage in app/config.py to the export date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
