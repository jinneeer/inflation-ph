"""Forecast scenario engine: SARIMA and ETS, fitted on national log CPI.

Baseline path comes from the published artifact; scenario paths are fitted
live (bounded parameters, LRU-cached). A price shock is translated with the
basket weight: Δheadline ≈ w_node × shock% (documented approximation).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from app.config import settings
from app.db import query
from app.services.models import WEIGHTS


@dataclass(frozen=True)
class ForecastParams:
    model: str = "sarima"          # sarima | ets | both
    p: int = 1
    d: int = 1
    q: int = 1
    P: int = 1
    D: int = 1
    Q: int = 0
    trend: str = "add"
    seasonal: str = "add"
    damped: bool = True
    horizon: int = 24
    shock_node: str | None = None
    shock_pct: float = 0.0
    shock_start: int = 1


def national_cpi(start: str | None = None) -> pd.DataFrame:
    """National CPI used to fit the models (default: 2018-base era, see settings.fit_start)."""
    return pd.DataFrame(query("headline", [start or settings.fit_start, "2026-10-01"]).rows)[["date", "cpi"]]


def _weight(node: str) -> float:
    row = WEIGHTS.loc[WEIGHTS["commodity_code"].astype(str) == str(node)]
    return float(row["weight_pct"].iloc[0]) / 100.0 if not row.empty else 0.0


def _shock_factor(node: str | None, pct: float, start: int, horizon: int) -> np.ndarray:
    factor = np.ones(horizon)
    if node and pct:
        w = _weight(node)
        level = max(0.01, 1.0 + w * pct / 100.0)  # weight-based approximation
        if start <= horizon:
            factor[start - 1:] = level
    return factor


def _yoy_path(cpi_hist: np.ndarray, path: np.ndarray) -> np.ndarray:
    ext = np.concatenate([cpi_hist, path])
    out = np.empty(len(path))
    for i in range(len(path)):
        cur = len(cpi_hist) + i
        out[i] = (ext[cur] / ext[cur - 12] - 1.0) * 100.0
    return out


@lru_cache(maxsize=32)
def _fit(model: str, p: int, d: int, q: int, P: int, D: int, Q: int,
         trend: str, seasonal: str, damped: bool, horizon: int,
         shock_node: str | None, shock_pct: float, shock_start: int) -> list[dict]:
    df = national_cpi()
    hist = df["cpi"].to_numpy()
    log_cpi = np.log(hist)
    dates = pd.date_range(pd.Timestamp(df["date"].max()) + pd.offsets.MonthBegin(1),
                          periods=horizon, freq="MS")
    shock = _shock_factor(shock_node, shock_pct, shock_start, horizon)
    out: list[dict] = []

    if model in ("sarima", "both"):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fit = SARIMAX(log_cpi, order=(p, d, q), seasonal_order=(P, D, Q, 12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.get_forecast(steps=horizon)
        mean = np.exp(np.asarray(fc.predicted_mean)) * shock
        ci80 = np.exp(np.asarray(fc.conf_int(alpha=0.20))) * shock[:, None]
        ci95 = np.exp(np.asarray(fc.conf_int(alpha=0.05))) * shock[:, None]
        out.append(_frame(dates, mean, ci80[:, 0], ci80[:, 1], ci95[:, 0], ci95[:, 1],
                          hist, "SARIMA", aic=round(float(fit.aic), 2)))

    if model in ("ets", "both"):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        if idx.inferred_freq == "MS":
            idx = pd.DatetimeIndex(idx, freq="MS")
        series = pd.Series(hist, index=idx)
        trend_arg = None if trend == "none" else trend
        seasonal_arg = None if seasonal == "none" else seasonal
        damped_arg = bool(damped and trend_arg is not None)
        fit = ETSModel(series, error="add", trend=trend_arg, damped_trend=damped_arg,
                       seasonal=seasonal_arg, seasonal_periods=12,
                       initialization_method="estimated").fit(disp=False)
        pred = fit.get_prediction(start=dates[0], end=dates[-1])
        summary = pred.summary_frame(alpha=0.20)
        mean = summary["mean"].to_numpy() * shock
        lo80 = summary["pi_lower"].to_numpy() * shock
        hi80 = summary["pi_upper"].to_numpy() * shock
        summary95 = pred.summary_frame(alpha=0.05)
        lo95 = summary95["pi_lower"].to_numpy() * shock
        hi95 = summary95["pi_upper"].to_numpy() * shock
        out.append(_frame(dates, mean, lo80, hi80, lo95, hi95, hist, "ETS",
                          aic=round(float(fit.aic), 2)))
    return out


def _frame(dates, mean, lo80, hi80, lo95, hi95, hist, label: str, aic: float | None = None) -> dict:
    yoy = _yoy_path(hist, mean)
    yoy80l = _yoy_path(hist, lo80)
    yoy80h = _yoy_path(hist, hi80)
    yoy95l = _yoy_path(hist, lo95)
    yoy95h = _yoy_path(hist, hi95)
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "model": label,
            "cpi": round(float(mean[i]), 3),
            "inflation": round(float(yoy[i]), 3),
            "lo80": round(float(yoy80l[i]), 3),
            "hi80": round(float(yoy80h[i]), 3),
            "lo95": round(float(yoy95l[i]), 3),
            "hi95": round(float(yoy95h[i]), 3),
        })
    return {
        "model": label,
        "rows": rows,
        "aic": aic,
    }


def forecast(params: ForecastParams) -> dict:
    paths = _fit(params.model, params.p, params.d, params.q,
                 params.P, params.D, params.Q, params.trend, params.seasonal,
                 params.damped, params.horizon, params.shock_node,
                 float(params.shock_pct), int(params.shock_start))
    df = national_cpi().copy()
    df["value"] = df["cpi"].pct_change(12) * 100.0
    tail = df.dropna(subset=["value"]).tail(72)
    history = [{"date": str(d)[:10], "value": round(float(v), 3)}
               for d, v in zip(tail["date"], tail["value"])]
    return {"history": history, "paths": paths,
            "notes": ["SARIMA/ETS intervals assume correct specification; structural breaks excluded.",
                      "Price shock uses the 2018 basket weight approximation."]}
