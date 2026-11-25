"""Pipeline, nowcast, story and rubric-critical behaviour."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import forecast as fc
from app.services import models

client = TestClient(app)
NCR = "National Capital Region (NCR)"


def test_feature_table_rebuilds_from_warehouse():
    from pipeline.build_features import build
    rebuilt = build()
    committed = pd.read_parquet(ROOT / "data" / "warehouse" / "feat_region_month.parquet")
    assert len(rebuilt) == len(committed)
    rebuilt["date"] = pd.to_datetime(rebuilt["date"])
    committed["date"] = pd.to_datetime(committed["date"])
    num = committed.select_dtypes("number").columns
    merged = committed.merge(rebuilt, on=["date", "geolocation"], suffixes=("", "_r"))
    for col in num:
        if col + "_r" in merged:
            diff = (merged[col].fillna(0) - merged[col + "_r"].fillna(0)).abs().max()
            assert diff < 1e-6, col


def test_live_sarima_defaults_match_published_artifact():
    """The default live fit must reproduce the stored forecast the paper cites."""
    assert fc.ForecastParams().D == 1
    live = client.post("/api/forecast", json={"model": "sarima", "horizon": 12}).json()["data"][0]["rows"]
    stored = models.FORECAST[models.FORECAST["model"] == "SARIMA"].sort_values("date")
    assert abs(live[0]["inflation"] - float(stored.iloc[0]["inflation"])) < 0.05
    assert abs(live[0]["hi95"] - float(stored.iloc[0]["infl_hi95"])) < 0.05


def test_national_kpi_does_not_average_regional_scores():
    nat = client.get("/api/kpis", params={"end": "2026-10-01"}).json()["data"][0]
    assert nat["breach_probability"] is None
    assert nat["regions_above_next"]["total"] == 18
    reg = client.get("/api/kpis", params={"end": "2026-10-01", "region": NCR}).json()["data"][0]
    assert 0 <= reg["breach_probability"] <= 1
    assert reg["nowcast_next"] == reg["headline"] or abs(reg["nowcast_next"] - reg["headline"]) < 0.06


def test_nowcast_whatif_changes_model_inputs():
    base = client.post("/api/nowcast/whatif", json={"region": NCR, "infl": 5.9}).json()["data"]
    low = client.post("/api/nowcast/whatif", json={"region": NCR, "infl": 2.0}).json()["data"]
    assert low["scenario_next"] < base["scenario_next"]
    assert low["scenario_proba"] < base["scenario_proba"]
    assert low["features"]["ma3"] < base["features"]["ma3"]      # derived inputs are recomputed
    assert client.post("/api/nowcast/whatif", json={"region": "nowhere", "infl": 3}).status_code == 404


def test_forecast_accepts_model_hyperparameters():
    a = client.post("/api/forecast", json={"model": "sarima", "horizon": 6, "p": 1, "q": 1}).json()
    b = client.post("/api/forecast", json={"model": "sarima", "horizon": 6, "p": 2, "q": 0, "Q": 1}).json()
    assert a["data"] and b["data"]
    assert a["data"][0]["rows"][3]["inflation"] != b["data"][0]["rows"][3]["inflation"]


def test_story_is_written_from_data():
    body = client.get("/api/story").json()["data"]
    ids = [c["id"] for c in body["chapters"]]
    assert ids == ["now", "where", "why", "next", "trust", "act"]
    head = client.get("/api/kpis").json()["data"][0]["headline"]
    assert f"{head:.1f}%" in body["chapters"][0]["bullets"][0]
    assert all(c["bullets"] for c in body["chapters"])
    assert len(body["glossary"]) >= 5


def test_honest_baseline_is_reported():
    m = client.get("/api/model/metrics").json()["data"]
    assert "headline_nowcast" in m and "classifier_baseline" in m
    for fold in m["nowcast"]["folds"]:
        assert "Persistence (this month carried forward)" in fold["regression"]
