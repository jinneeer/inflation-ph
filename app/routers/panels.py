from __future__ import annotations

from fastapi import APIRouter, Query

from app import db
from app.services import models

router = APIRouter()


def envelope(result: db.QueryResult, extra: dict | None = None) -> dict:
    meta = result.meta()
    if extra:
        meta.update(extra)
    return {"data": result.rows, "meta": meta}


@router.get("/kpis")
def kpis(end: str = Query("2026-10-01"), region: str | None = Query(None)):
    regional = bool(region and region != "PHILIPPINES")
    if regional:
        res = db.query("kpis_region", [region, end, region, region])
        row = res.rows[0] if res.rows else {}
        nat = db.query("kpis", [end]).rows
        national = nat[0] if nat else {}
        row["core"] = national.get("core")
        nc = models.latest_nowcast(region, end)
        breach = nc["proba"] if nc else None
        nowcast = nc["pred_next"] if nc else None
        extra = {"nowcast_lo80": nc["lo80"], "nowcast_hi80": nc["hi80"]} if nc else {}
        panel = "kpis_region"
    else:
        res = db.query("kpis", [end])
        row = res.rows[0] if res.rows else {}
        # the breach model is regional: report how many regions it expects above 4%, not a made-up national odds
        breach = None
        extra = {"regions_above_next": models.regions_expected_above(end)}
        fc = models.FORECAST[models.FORECAST["model"] == "SARIMA"]
        nowcast = float(fc.sort_values("date").iloc[0]["inflation"]) if not fc.empty else None
        panel = "kpis"
    row.update({
        "region": region or "PHILIPPINES",
        "status": "above_band" if (row.get("headline") or 0) > 4 else "within_band",
        "nowcast_next": nowcast,
        "breach_probability": breach,
        "bsp_band": [2.0, 4.0],
        **extra,
    })
    meta = res.meta()
    meta["panel_id"] = panel
    return {"data": [row], "meta": meta}


@router.get("/series/headline")
def headline(start: str = Query("2018-01-01"), end: str = Query("2026-10-01")):
    return envelope(db.query("headline", [start, end]))


@router.get("/series/region")
def region_series(region: str = Query(...), start: str = Query("2018-01-01"),
                  end: str = Query("2026-10-01")):
    return envelope(db.query("region_series", [region, start, end]))


@router.get("/map/choropleth")
def choropleth(date: str = Query("2026-10-01")):
    return envelope(db.query("map", [date]), {"metric": "yoy"})


@router.get("/drivers/treemap")
def treemap(date: str = Query("2026-10-01"), region: str = Query("PHILIPPINES")):
    return envelope(db.query("treemap", [date, region]))


@router.get("/drivers/heatmap")
def sector_heatmap(end: str = Query("2026-10-01")):
    return envelope(db.query("sector_heatmap", [end, end]))


@router.get("/calendar")
def calendar(start: str = Query("2018-01-01"), end: str = Query("2026-10-01")):
    return envelope(db.query("calendar", [start, end]))


@router.get("/distribution")
def distribution():
    return envelope(db.query("distribution", []))


@router.get("/table/geolocations")
def geolocations():
    return envelope(db.query("region_table", []))
