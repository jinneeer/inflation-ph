"""Forecast Lab: SARIMA/ETS fan charts (80% / 95%), horizon + confidence + smoothing + order controls, price-shock what-if, backtest."""
from __future__ import annotations
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from scipy.stats import norm
from src import common as C, forecast as F

GROUPS_SQL = """SELECT i.item_code, w.weight_national AS weight FROM fact_weight w JOIN dim_region r USING (region_id)
JOIN dim_item i USING (item_id) WHERE r.is_national AND NOT i.is_headline ORDER BY 1"""


def render():
    st.header("Forecast lab — where is national inflation heading?")
    st.caption("Philippines-wide headline inflation. Change the controls; the chart refits live. Shaded bands = how sure the model is.")
    cpi = C.national_cpi()
    yoy_all = ((cpi / cpi.shift(12) - 1) * 100).dropna()
    a, b, c, d = st.columns(4)
    model = a.radio("Model", ["sarima", "ets"], format_func=str.upper, horizontal=True)
    horizon = b.slider("Months ahead", 3, 24, 12)
    level = c.radio("Uncertainty band", ["80%", "95%", "Both"], index=2, horizontal=True)
    k = d.slider("Smoothing of history (k months)", 1, 12, 1, help="Moving average over k months; 1 = raw data")
    with st.expander("Model settings, forecast origin and price what-if", expanded=False):
        e1, e2, e3, e4 = st.columns(4)
        p = e1.slider("AR order p", 0, 2, 1); dd = e2.slider("Differencing d", 0, 1, 1); qq = e3.slider("MA order q", 0, 2, 1)
        seas = e4.checkbox("Seasonal pattern (12-month)", True)
        origins = [m for m in C.months() if m >= pd.Timestamp("2022-01-01")]
        origin = st.select_slider("Pretend it is… (forecast origin; compare with what really happened)", options=origins, value=origins[-1], format_func=C.fmt_month)
        w = C.q("National basket weights", GROUPS_SQL).set_index("item_code")["weight"]
        f1, f2, f3 = st.columns(3)
        grp = f1.selectbox("Price shock group", [g for g in w.index], format_func=lambda x: C.SHORT[x], index=list(w.index).index("07"))
        shock = f2.slider("Price change (%)", -20, 30, 0)
        lag = f3.slider("Starts in … months", 1, 6, 1)
    order, seasonal = (p, dd, qq), ((1, 1, 0, 12) if seas else (0, 0, 0, 0))
    if (p, dd, qq) == (0, 0, 0) and not seas:
        st.warning("That setting is a flat line; add an AR/MA term."); return
    up = origin.strftime("%Y-%m-%d")
    fc = C.national_forecast(model, horizon, order, seasonal, up)
    hist = yoy_all.loc[:origin]
    sm = hist.rolling(k, min_periods=1).mean() if k > 1 else hist
    fig = go.Figure(); C.target_band(fig)
    levels = {"80%": [80], "95%": [95], "Both": [95, 80]}[level]
    for lv, col in [(95, "rgba(14,116,144,0.15)"), (80, "rgba(14,116,144,0.32)")]:
        if lv in levels:
            fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc[f"yoy_hi{lv}"]) + list(fc[f"yoy_lo{lv}"][::-1]), fill="toself",
                            fillcolor=col, line=dict(width=0), name=f"{lv}% band", hoverinfo="skip")
    fig.add_scatter(x=sm.index, y=sm.values, mode="lines", line=dict(color=C.BLACK, width=2.5), name=f"History{' (k=%d)' % k if k > 1 else ''}",
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>")
    if origin < cpi.index[-1]:
        real = yoy_all.loc[origin:]
        fig.add_scatter(x=real.index, y=real.values, mode="lines", line=dict(color=C.GREEN, width=2.5, dash="dot"), name="What really happened",
                        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>actual</extra>")
    fig.add_scatter(x=fc.index, y=fc["yoy"], mode="lines+markers", line=dict(color=C.BLUE, width=3), name="Forecast",
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>forecast</extra>")
    if shock:
        sf = F.apply_shock(fc, cpi.loc[:up], float(w[grp]), shock, lag)
        fig.add_scatter(x=sf.index, y=sf["yoy"], mode="lines", line=dict(color=C.VERMIL, width=3, dash="dash"),
                        name=f"What-if: {C.SHORT[grp]} {shock:+d}%", hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>what-if</extra>")
    fig.update_yaxes(ticksuffix="%", title="% change vs a year earlier")
    st.plotly_chart(C.style(fig, f"Fan chart from {C.fmt_month(origin)}: {model.upper()} {order}{seasonal if seas else ''}", 480), width="stretch")

    sd = ((fc["yoy_hi80"] - fc["yoy_lo80"]) / (2 * 1.2816)).clip(lower=1e-6)
    pb = 1 - norm.cdf((C.TARGET_HI - fc["yoy"]) / sd)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Next month", f"{fc['yoy'].iloc[0]:.1f}%", f"80% range {fc['yoy_lo80'].iloc[0]:.1f}–{fc['yoy_hi80'].iloc[0]:.1f}")
    m2.metric(f"In {horizon} months", f"{fc['yoy'].iloc[-1]:.1f}%", f"80% range {fc['yoy_lo80'].iloc[-1]:.1f}–{fc['yoy_hi80'].iloc[-1]:.1f}", delta_color="off")
    m3.metric("Highest expected", f"{fc['yoy'].max():.1f}%", C.fmt_month(fc["yoy"].idxmax()), delta_color="off")
    m4.metric(f"Chance above 4% in {horizon} mo", f"{100 * pb[-1]:.0f}%")
    if shock:
        st.info(f"What-if: a {shock:+d}% change in **{C.SHORT[grp]}** ({w[grp]:.1f}% of the basket) lifts the forecast by about "
                f"**{w[grp] / 100 * shock:+.2f} pts** from month {lag} on. First-order approximation; other prices held fixed.")
    C.sql_box("National basket weights")

    st.subheader("How good have these forecasts been? (rolling-origin backtest)")
    bt, _ = C.backtests()
    bt = bt.assign(bucket=pd.cut(bt["h"], [0, 3, 6, 12], labels=["1–3 mo", "4–6 mo", "7–12 mo"]))
    tab = bt.pivot_table(index="bucket", columns="model", values="abs_err", aggfunc="mean", observed=True).round(2)
    cov = bt.dropna(subset=["in80"]).groupby("model")["in80"].mean().mul(100).round(0)
    l, r = st.columns([1.2, 1])
    fig2 = go.Figure()
    for mdl, col in [("carry-forward", C.ORANGE), ("sarima", C.BLUE), ("ets", C.PINK)]:
        t = bt[bt["model"] == mdl].groupby("h")["abs_err"].mean()
        fig2.add_scatter(x=t.index, y=t.values, mode="lines+markers", name={"carry-forward": "Carry this month forward"}.get(mdl, mdl.upper()),
                         line=dict(color=col, width=2.5), hovertemplate="%{x} mo ahead: miss %{y:.2f} pts<extra></extra>")
    fig2.update_xaxes(title="months ahead"); fig2.update_yaxes(title="average miss (pts of inflation)")
    l.plotly_chart(C.style(fig2, "Average miss by horizon (origins 2023-10, 2024-10, 2025-10, 2026-04)", 320), width="stretch")
    r.dataframe(tab.rename(columns=str.upper), width="stretch")
    r.caption(f"Share of real outcomes inside the 80% band — SARIMA {cov.get('sarima', float('nan')):.0f}%, ETS {cov.get('ets', float('nan')):.0f}% (target 80%). "
              "**Honest read:** simply carrying this month forward is hard to beat for 1–3 months; the models earn their keep by giving ranges and what-ifs.")
