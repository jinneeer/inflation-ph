"""Persisted model artifacts, read once and cached."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
GEO = ROOT / "data" / "geo"


def _read(name: str) -> pd.DataFrame:
    path = MODELS / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


METRICS = json.loads((MODELS / "metrics.json").read_text()) if (MODELS / "metrics.json").exists() else {}
ROC = _read("roc_curve.parquet")
PR = _read("pr_curve.parquet")
CLF = _read("clf_test.parquet")
SHAP = _read("shap_values.parquet")
FEATURES = _read("feature_importance.parquet")
PCA = _read("pca_regions.parquet")
FORECAST = _read("forecast_national.parquet")
BACKTEST = _read("backtest.parquet")
NOWCAST_TEST = _read("nowcast_test.parquet")
NOWCAST_LATEST = _read("nowcast_latest.parquet")

WEIGHTS = pd.read_csv(ROOT / "data" / "cpi_weights_2018base_clean.csv")
GEOJSON_TEXT = (GEO / "ph_regions.geojson").read_text(encoding="utf-8")
GEO_KEY = pd.read_csv(GEO / "geo_key_map.csv")


def local_shap(region: str, date: str) -> list[dict]:
    """Stored per-row SHAP contributions for one region-month (≤ 12 features)."""
    if SHAP.empty:
        return []
    df = SHAP[(SHAP["geolocation"] == region) &
              (pd.to_datetime(SHAP["date"]) == pd.Timestamp(date))]
    if df.empty:
        return []
    df = df.sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
    return [{"feature": r.feature, "value": float(r.feature_value),
             "contribution": float(r.shap_value)} for r in df.itertuples()]


def latest_nowcast(region: str, date: str | None = None) -> dict | None:
    """Next-month view for one region: carried-forward rate, model adjustment, breach probability.

    ``date`` is the last observed month. The newest month comes from nowcast_latest; months
    inside the held-out test window come from nowcast_test / clf_test; anything else is None.
    """
    if not NOWCAST_LATEST.empty:
        row = NOWCAST_LATEST[NOWCAST_LATEST["geolocation"] == region]
        if not row.empty and (date is None or pd.Timestamp(date) >= pd.Timestamp(row.iloc[0]["date"])):
            r = row.iloc[0]
            return {"date": str(r["date"])[:10], "infl_now": float(r["infl_now"]),
                    "pred_next": float(r["pred_next"]), "model_pred": float(r["model_pred"]),
                    "lo80": float(r["lo80"]), "hi80": float(r["hi80"]),
                    "proba": float(r["proba_above4"])}
    if date and not NOWCAST_TEST.empty:
        t = NOWCAST_TEST[(NOWCAST_TEST["geolocation"] == region) &
                         (pd.to_datetime(NOWCAST_TEST["date"]) == pd.Timestamp(date))]
        if not t.empty:
            r = t.iloc[0]
            c = CLF[(CLF["geolocation"] == region) & (pd.to_datetime(CLF["date"]) == pd.Timestamp(date))]
            return {"date": str(date)[:10], "infl_now": float(r["infl_now"]),
                    "pred_next": float(r["pred"]), "model_pred": float(r["model_pred"]),
                    "lo80": float(r["lo80"]), "hi80": float(r["hi80"]),
                    "proba": float(c.iloc[0]["proba"]) if not c.empty else None}
    return None


def regions_expected_above(date: str | None = None, cut: float = 0.5) -> dict:
    """How many regions the breach model expects above 4% next month."""
    if NOWCAST_LATEST.empty:
        return {"count": None, "total": 0}
    df = NOWCAST_LATEST
    if date and pd.Timestamp(date) < pd.Timestamp(df["date"].iloc[0]):
        if CLF.empty:
            return {"count": None, "total": 0}
        df = CLF[pd.to_datetime(CLF["date"]) == pd.Timestamp(date)].rename(columns={"proba": "proba_above4"})
        if df.empty:
            return {"count": None, "total": 0}
    return {"count": int((df["proba_above4"] >= cut).sum()), "total": int(len(df))}


def classifier_metrics(threshold: float) -> dict:
    if CLF.empty:
        return {}
    pred = (CLF["proba"] >= threshold).astype(int)
    tp = int(((pred == 1) & (CLF["y_true"] == 1)).sum())
    fp = int(((pred == 1) & (CLF["y_true"] == 0)).sum())
    fn = int(((pred == 0) & (CLF["y_true"] == 1)).sum())
    tn = int(((pred == 0) & (CLF["y_true"] == 0)).sum())
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / len(CLF),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(CLF),
    }
