"""API contract and aggregation-guard tests."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def get(path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text[:300]}"
    body = r.json()
    if "data" in body and isinstance(body["data"], list):
        assert len(body["data"]) < 1000, f"{path} returned {len(body['data'])} rows"
    return body


def test_health():
    assert get("/api/health")["status"] == "ok"


def test_meta():
    meta = get("/api/meta")
    assert meta["bsp_band"] == [2.0, 4.0]
    assert len(meta["regions"]) == 18
    assert meta["date_bounds"]["end"] == "2026-10-01"


def test_kpis():
    rows = get("/api/kpis", end="2026-10-01")["data"]
    assert rows and rows[0]["headline"] > 0
    assert rows[0]["status"] in ("above_band", "within_band")
    reg = get("/api/kpis", end="2026-10-01",
              region="National Capital Region (NCR)")["data"][0]
    assert reg["headline"] != rows[0]["headline"]
    assert reg["cpi"] > 0


def test_headline_and_region():
    h = get("/api/series/headline", start="2018-01-01", end="2026-10-01")["data"]
    assert 100 <= len(h) <= 110
    r = get("/api/series/region", region="National Capital Region (NCR)",
            start="2018-01-01", end="2026-10-01")["data"]
    assert len(r) > 50


def test_map_and_tables():
    m = get("/api/map/choropleth", date="2026-10-01")["data"]
    assert len(m) == 18
    assert {"value", "prev", "mom_change", "rank"} <= set(m[0])
    t = get("/api/table/geolocations")["data"]
    assert len(t) == 18


def test_drivers_and_calendar():
    tr = get("/api/drivers/treemap", date="2026-10-01")["data"]
    assert len(tr) == 13
    heat = get("/api/drivers/heatmap", end="2026-10-01")["data"]
    assert len(heat) < 1000
    cal = get("/api/calendar", start="2018-01-01", end="2026-10-01")["data"]
    assert len(cal) < 1000
    dist = get("/api/distribution")["data"]
    assert len(dist) > 10


def test_forecast_sarima():
    body = client.post("/api/forecast", json={"model": "sarima", "horizon": 12}).json()
    assert body["data"], body
    rows = body["data"][0]["rows"]
    assert len(rows) == 12
    assert all(set(r) >= {"date", "inflation", "lo80", "hi80", "lo95", "hi95"} for r in rows)


def test_forecast_ets_and_shock():
    body = client.post("/api/forecast", json={
        "model": "ets", "horizon": 9, "shock_node": "01.1.1.12",
        "shock_pct": 20, "shock_start": 2,
    }).json()
    assert body["data"], body
    rows = body["data"][0]["rows"]
    assert len(rows) == 9
    assert body["baseline"], "no-shock baseline missing"
    assert body["meta"]["fit_ms"] > 0


def test_local_shap():
    body = get("/api/nowcast/shap/local",
               region="National Capital Region (NCR)", date="2026-09-01")
    assert len(body["data"]) == 13
    assert body["base"] is not None
    total = body["start"] + body["base"] + sum(r["contribution"] for r in body["data"])
    assert abs(total - body["prediction"]) < 0.01


def test_diagnostics():
    c = get("/api/classifier/curves")["data"]
    assert c["auc"] > 0.8
    th = client.post("/api/classifier/threshold", json={"threshold": 0.5}).json()["data"]
    assert 0 <= th["precision"] <= 1
    g = get("/api/nowcast/shap/global")["data"]
    assert len(g) == 13
    b = get("/api/nowcast/shap/beeswarm", sample=40)["data"]
    assert 0 < len(b) < 1000
    p = get("/api/structure/pca")["data"]
    assert len(p) == 18
    res = get("/api/nowcast/residuals")["data"]
    assert len(res) == 198 and {"geolocation", "actual", "pred", "residual"} <= set(res[0])


def test_sql_preview_and_export():
    sql = get("/api/sql/map")["data"]["sql"]
    assert "fact_inflation" in sql
    r = client.get("/api/export/map.csv", params={"date": "2026-10-01"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")


def test_edge_cases():
    ets_none = client.post("/api/forecast", json={
        "model": "ets", "trend": "none", "seasonal": "none", "damped": True, "horizon": 6,
    }).json()
    assert ets_none["data"] and len(ets_none["data"][0]["rows"]) == 6
    assert ets_none["data"][0]["aic"] is not None

    bad_news = client.get("/api/news", params={"month": "2026-13-01"})
    assert bad_news.status_code == 422

    bad_export = client.get("/api/export/unknown_panel.csv")
    assert bad_export.status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("ALL API TESTS PASSED")
