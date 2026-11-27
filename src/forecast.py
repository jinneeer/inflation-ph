"""National headline-CPI forecasts with 80% / 95% prediction intervals.

SARIMA and ETS are fitted on log CPI; the yoy path is rebuilt from the CPI path, so a price
shock applied to the CPI level (what-if) flows straight through to inflation.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
LEVELS = (80, 95)


def national_cpi(con) -> pd.Series:
    df = con.execute("""
        SELECT d.month_start, f.cpi FROM fact_cpi f
        JOIN dim_region r USING (region_id) JOIN dim_item i USING (item_id) JOIN dim_date d USING (date_id)
        WHERE r.is_national AND i.is_headline ORDER BY 1""").df()
    s = pd.Series(df["cpi"].values, index=pd.DatetimeIndex(df["month_start"], freq="MS"), name="cpi")
    return s


def _future_index(last: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def _sarima(ylog, h, order, seasonal):
    res = SARIMAX(ylog, order=order, seasonal_order=seasonal, enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False, maxiter=200)
    fc = res.get_forecast(h)
    out = pd.DataFrame({"mean": np.asarray(fc.predicted_mean)})
    for lv in LEVELS:
        ci = np.asarray(fc.conf_int(alpha=1 - lv / 100))
        out[f"lo{lv}"], out[f"hi{lv}"] = ci[:, 0], ci[:, 1]
    return out


def _ets(ylog, h):
    res = ETSModel(ylog, error="add", trend="add", damped_trend=True, seasonal="add",
                   seasonal_periods=12).fit(disp=False)
    n = len(ylog)
    out = None
    for lv in LEVELS:
        sf = res.get_prediction(start=n, end=n + h - 1).summary_frame(alpha=1 - lv / 100)
        if out is None:
            out = pd.DataFrame({"mean": sf["mean"].values})
        out[f"lo{lv}"], out[f"hi{lv}"] = sf["pi_lower"].values, sf["pi_upper"].values
    return out


def forecast(cpi: pd.Series, horizon: int = 12, model: str = "sarima",
             order=(1, 1, 1), seasonal=(1, 1, 0, 12)) -> pd.DataFrame:
    """Return CPI and yoy(%) forecasts with both interval levels. Index = future months."""
    ylog = np.log(cpi)
    raw = _sarima(ylog, horizon, order, seasonal) if model == "sarima" else _ets(ylog, horizon)
    idx = _future_index(cpi.index[-1], horizon)
    out = pd.DataFrame(index=idx)
    out["cpi"] = np.exp(raw["mean"].values)
    for lv in LEVELS:
        out[f"cpi_lo{lv}"], out[f"cpi_hi{lv}"] = np.exp(raw[f"lo{lv}"].values), np.exp(raw[f"hi{lv}"].values)
    ext = pd.concat([cpi, out["cpi"]])
    base = ext.shift(12).loc[idx].values          # same month last year (actual, or forecast beyond 12 months)
    out["yoy"] = (out["cpi"].values / base - 1) * 100
    for lv in LEVELS:
        out[f"yoy_lo{lv}"] = (out[f"cpi_lo{lv}"].values / base - 1) * 100
        out[f"yoy_hi{lv}"] = (out[f"cpi_hi{lv}"].values / base - 1) * 100
    return out


def apply_shock(fc: pd.DataFrame, cpi: pd.Series, weight_pct: float, shock_pct: float, start_lag: int = 1) -> pd.DataFrame:
    """What-if: a `shock_pct` price rise in a group worth `weight_pct` of the basket lifts headline CPI by
    ~ weight * shock from `start_lag` months ahead (first-order approximation, other prices unchanged)."""
    f = 1 + (weight_pct / 100) * (shock_pct / 100)
    out = fc.copy()
    on = np.arange(len(out)) >= (start_lag - 1)
    mult = np.where(on, f, 1.0)
    for c in [c for c in out.columns if c.startswith("cpi")]:
        out[c] = out[c].values * mult
    ext = pd.concat([cpi, out["cpi"]])
    base = ext.shift(12).loc[out.index].values
    # base must stay the *unshocked* history/forecast for months beyond 12; use unshocked ext for shifted forecast base
    base = pd.concat([cpi, fc["cpi"]]).shift(12).loc[out.index].values
    out["yoy"] = (out["cpi"].values / base - 1) * 100
    for lv in LEVELS:
        out[f"yoy_lo{lv}"] = (out[f"cpi_lo{lv}"].values / base - 1) * 100
        out[f"yoy_hi{lv}"] = (out[f"cpi_hi{lv}"].values / base - 1) * 100
    return out


def rolling_origin(cpi: pd.Series, origins: list[str], horizon: int = 12) -> pd.DataFrame:
    """Expanding-window backtest: fit up to each origin, forecast `horizon` months, score against actuals."""
    actual_yoy = (cpi / cpi.shift(12) - 1) * 100
    rows = []
    for o in origins:
        o = pd.Timestamp(o)
        train = cpi.loc[:o]
        fut = _future_index(o, horizon)
        fut = fut[fut <= cpi.index[-1]]
        h = len(fut)
        if h == 0:
            continue
        persist = actual_yoy.loc[o]
        for model in ("sarima", "ets"):
            fc = forecast(train, h, model)
            for k, d in enumerate(fut, 1):
                rows.append((model, o, k, d, actual_yoy.loc[d], fc["yoy"].iloc[k - 1],
                             fc["yoy_lo80"].iloc[k - 1], fc["yoy_hi80"].iloc[k - 1]))
        for k, d in enumerate(fut, 1):
            rows.append(("carry-forward", o, k, d, actual_yoy.loc[d], persist, np.nan, np.nan))
    df = pd.DataFrame(rows, columns=["model", "origin", "h", "target", "actual", "pred", "lo80", "hi80"])
    df["abs_err"] = (df["pred"] - df["actual"]).abs()
    df["in80"] = ((df["actual"] >= df["lo80"]) & (df["actual"] <= df["hi80"])).where(df["lo80"].notna())
    return df
