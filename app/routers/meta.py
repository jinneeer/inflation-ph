from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, Response

from app import db
from app.services import models

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "data_vintage": db.DATA_VINTAGE}


@router.get("/meta")
def meta():
    gk = models.GEO_KEY.copy()
    gk["psgc_code"] = gk["psgc_code"].fillna("")
    regions = gk.to_dict("records")
    bounds = db.query("kpis", ["2026-10-01"]).rows[0]
    w = models.WEIGHTS.copy()
    top = w[w["commodity_code"].astype(str).str.fullmatch(r"\d+") &
            (w["commodity_code"].astype(str) != "0")]
    sectors = [{"code": str(r.commodity_code), "name": r.commodity.title()}
               for r in top.itertuples()]
    # everyday items first, so a non-technical user can try "what if fuel or rice rises?" at once
    curated = {"07.2.2": "Fuel for vehicles", "04.5.1": "Electricity", "04.5.2.2": "Cooking gas (LPG)",
               "01.1.1.12": "Rice", "01.1.2.2": "Meat", "01.1.3.1": "Fish", "01.1": "All food"}
    codes = w["commodity_code"].astype(str)
    nodes = w[codes.str.count(r"\.") >= 1]
    pick = [nodes[nodes["commodity_code"].astype(str) == c].iloc[0] for c in curated
            if (nodes["commodity_code"].astype(str) == c).any()]
    rest = [r for _, r in nodes.head(40).iterrows() if str(r["commodity_code"]) not in curated]
    shock_nodes = [{"code": str(r["commodity_code"]),
                    "name": curated.get(str(r["commodity_code"]), str(r["commodity"]).title()),
                    "weight": round(float(r["weight_pct"]), 3)} for r in pick + rest]
    return {
        "regions": regions,
        "sectors": sectors,
        "shock_nodes": shock_nodes,
        "date_bounds": {"start": "2018-01-01", "end": "2026-10-01"},
        "bsp_band": [2.0, 4.0],
        "data_vintage": db.DATA_VINTAGE,
        "model_version": models.METRICS.get("trained_at"),
        "default": {"end": str(bounds["date"])[:10], "headline": bounds["headline"]},
    }


@router.get("/sql/{panel_id}")
def sql_preview(panel_id: str):
    try:
        text = db.sql_text(panel_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown panel {panel_id}")
    return {"data": {"sql": text}, "meta": {"panel_id": panel_id}}


@router.get("/export/{panel_id}.csv")
def export_csv(panel_id: str, response: Response,
               start: str = Query("2018-01-01"), end: str = Query("2026-10-01"),
               region: str | None = None, date: str | None = None):
    params = {
        "headline": [start, end],
        "calendar": [start, end],
        "distribution": [],
        "kpis": [end],
        "map": [date or end],
        "treemap": [date or end, region or "PHILIPPINES"],
        "region_table": [],
        "sector_heatmap": [end, end],
        "region_series": [region or "National Capital Region (NCR)", start, end],
    }.get(panel_id)
    if params is None:
        raise HTTPException(status_code=404, detail="not exportable")
    result = db.query(panel_id, params)
    import csv
    import io
    buf = io.StringIO()
    if result.rows:
        writer = csv.DictWriter(buf, fieldnames=list(result.rows[0].keys()))
        writer.writeheader()
        writer.writerows(result.rows)
    response.media_type = "text/csv"
    response.headers["Content-Disposition"] = f'attachment; filename="{panel_id}.csv"'
    return Response(content=buf.getvalue(), media_type="text/csv")


@router.get("/structure/pca")
def pca():
    rows = json.loads(models.PCA.to_json(orient="records"))
    return {"data": rows, "meta": {"panel_id": "structure_pca", "row_count": len(rows),
                                   "variance": models.METRICS.get("pca_explained"),
                                   "cache": "artifact"}}
