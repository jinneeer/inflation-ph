from __future__ import annotations

import dataclasses
import time

from fastapi import APIRouter, HTTPException

from app.schemas import ForecastRequest, NowcastWhatIf, ThresholdRequest
from app.services import forecast as svc
from app.services import models

router = APIRouter()


@router.post("/forecast")
def run_forecast(req: ForecastRequest):
    params = svc.ForecastParams(
        model=req.model, p=req.p, d=req.d, q=req.q, P=req.P, D=req.D, Q=req.Q,
        trend=req.trend, seasonal=req.seasonal, damped=req.damped,
        horizon=req.horizon, shock_node=req.shock_node,
        shock_pct=req.shock_pct, shock_start=req.shock_start,
    )
    t0 = time.perf_counter()
    try:
        result = svc.forecast(params)
        baseline = None
        if req.shock_node and req.shock_pct:
            base_params = dataclasses.replace(params, shock_node=None, shock_pct=0.0)
            baseline = svc.forecast(base_params)["paths"]
    except Exception as exc:  # keep the API honest about failed fits
        return {"data": [], "meta": {"panel_id": "forecast", "error": str(exc),
                                     "row_count": 0, "cache": "miss"}}
    fit_ms = round((time.perf_counter() - t0) * 1000, 1)
    rows = sum(len(p["rows"]) for p in result["paths"])
    return {"data": result["paths"], "history": result["history"], "baseline": baseline,
            "meta": {"panel_id": "forecast", "row_count": rows, "fit_ms": fit_ms,
                     "cache": "lru", "notes": result["notes"], "params": req.model_dump()}}


@router.get("/classifier/curves")
def curves():
    return {
        "data": {
            "roc": models.ROC.to_dict("records"),
            "pr": models.PR.to_dict("records"),
            "baseline": models.METRICS.get("classifier_baseline"),
            "base_rate": models.METRICS.get("classifier", {}).get("positive_rate"),
            "auc": models.METRICS.get("classifier", {}).get("AUC"),
            "ap": models.METRICS.get("classifier", {}).get("average_precision"),
        },
        "meta": {"panel_id": "classifier_curves"},
    }


@router.post("/classifier/threshold")
def threshold(req: ThresholdRequest):
    return {"data": models.classifier_metrics(req.threshold),
            "meta": {"panel_id": "classifier_threshold"}}


@router.get("/nowcast/shap/global")
def shap_global():
    rows = models.FEATURES.sort_values("mean_abs_shap").to_dict("records")
    return {"data": rows, "meta": {"panel_id": "shap_global", "row_count": len(rows)}}


@router.get("/nowcast/shap/beeswarm")
def shap_beeswarm(sample: int = 60):
    df = models.SHAP
    if df.empty:
        return {"data": [], "meta": {"panel_id": "shap_beeswarm", "row_count": 0}}
    # stride-sample rows while keeping every feature per chosen row
    rows = df.drop_duplicates(subset=["date", "geolocation"]).head(400)
    if len(rows) > sample:
        rows = rows.iloc[:: max(1, len(rows) // sample)]
    pick = set(zip(rows["date"], rows["geolocation"]))
    sub = df[[(d, g) in pick for d, g in zip(df["date"], df["geolocation"])]]
    if len(sub) > 999:
        sub = sub.iloc[:999]
    return {"data": sub.to_dict("records"),
            "meta": {"panel_id": "shap_beeswarm", "row_count": len(sub), "cache": "artifact"}}


@router.get("/nowcast/shap/local")
def shap_local(region: str, date: str):
    rows = models.local_shap(region, date)
    base = models.METRICS.get("shap_base_value")
    now = next((r["value"] for r in rows if r["feature"] == "infl"), None)
    # the model explains the *change* from this month: next = this month + base + contributions
    pred = round(now + base + sum(r["contribution"] for r in rows), 4) \
        if rows and base is not None and now is not None else None
    return {"data": rows, "prediction": pred, "base": base, "start": now,
            "meta": {"panel_id": "shap_local", "row_count": len(rows), "cache": "artifact"}}


@router.get("/nowcast/residuals")
def residuals(region: str | None = None):
    """Held-out regional misses (actual - carried-forward estimate), read from the stored artifact."""
    df = models.NOWCAST_TEST
    if df.empty:
        return {"data": [], "meta": {"panel_id": "nowcast_residuals", "row_count": 0}}
    if region and region != "PHILIPPINES":
        df = df[df["geolocation"] == region]
    rows = [{"date": str(r.date)[:10], "geolocation": r.geolocation,
             "actual": round(float(r.actual), 2), "pred": round(float(r.pred), 2),
             "model_pred": round(float(r.model_pred), 2), "residual": round(float(r.residual), 3)}
            for r in df.itertuples()]
    return {"data": rows, "meta": {"panel_id": "nowcast_residuals", "row_count": len(rows),
                                   "cache": "artifact"}}


@router.get("/nowcast/latest")
def nowcast_latest():
    df = models.NOWCAST_LATEST
    rows = df.assign(date=df["date"].astype(str).str[:10]).round(3).to_dict("records") if not df.empty else []
    return {"data": rows, "meta": {"panel_id": "nowcast_latest", "row_count": len(rows), "cache": "artifact"}}


@router.post("/nowcast/whatif")
def nowcast_whatif(req: NowcastWhatIf):
    from app.services.nowcast_model import nowcaster
    res = nowcaster().what_if(req.region, req.infl)
    if res is None:
        raise HTTPException(status_code=404, detail=f"unknown region {req.region}")
    return {"data": res, "meta": {"panel_id": "nowcast_whatif", "cache": "computed"}}


@router.get("/model/metrics")
def model_metrics():
    return {"data": models.METRICS, "meta": {"panel_id": "model_metrics"}}


@router.get("/model/backtest")
def model_backtest():
    rows = models.BACKTEST.round(3).to_dict("records")
    return {"data": rows, "meta": {"panel_id": "model_backtest", "row_count": len(rows)}}
