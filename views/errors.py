"""Where the model misses: click or lasso a high-error point and every connected view re-filters."""
from __future__ import annotations
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from src import common as C, nowcast as N


def render():
    st.header("Where does the model miss?")
    st.caption("Each dot is one region-month the model never saw. **Click a dot, or drag a box/lasso**, to filter the three views underneath.")
    s = C.q("Out-of-sample region-months", """SELECT region_name, region_short, month_start, yoy_all, pred_yoy, target_yoy, error, p_breach FROM nowcast_scored
        WHERE is_test ORDER BY month_start, region_name""")
    s["abs_err"] = s["error"].abs()
    only = st.checkbox("Only the selected region", False) and st.session_state["region"] != "Philippines"
    if only:
        s = s[s["region_name"] == st.session_state["region"]]
    s = s.reset_index(drop=True)
    lo, hi = float(min(s["target_yoy"].min(), s["pred_yoy"].min())) - 0.5, float(max(s["target_yoy"].max(), s["pred_yoy"].max())) + 0.5
    fig = go.Figure(go.Scatter(x=s["target_yoy"], y=s["pred_yoy"], mode="markers", customdata=np.c_[s["region_short"], s["month_start"].dt.strftime("%b %Y"), s["error"], s["yoy_all"]],
                               marker=dict(size=9 + 3 * s["abs_err"].clip(upper=3), color=s["abs_err"], colorscale=C.SEQUENTIAL, showscale=True, colorbar=dict(title="Miss<br>(pts)"),
                                           line=dict(color="white", width=0.7)),
                               hovertemplate="%{customdata[0]} · %{customdata[1]}<br>Actual %{x:.1f}% · Predicted %{y:.1f}%<br>Miss %{customdata[2]:+.2f} pts<extra></extra>"))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#888", dash="dot"), hoverinfo="skip", showlegend=False)
    fig.update_xaxes(title="What inflation actually was next month (%)", range=[lo, hi]); fig.update_yaxes(title="What the model predicted (%)", range=[lo, hi])
    ev = st.plotly_chart(C.style(fig, f"Predicted vs actual · {len(s)} region-months · typical miss {s['abs_err'].mean():.2f} pts", 470, legend=False),
                         width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"), key="err_scatter")
    C.sql_box("Out-of-sample region-months")
    idx = [p["point_index"] for p in (ev.selection.get("points", []) if ev and getattr(ev, "selection", None) else []) if p.get("point_index") is not None and p["point_index"] < len(s)]
    if not idx:
        idx = [int(s["abs_err"].idxmax())]
        st.info("Nothing selected yet — showing the **single biggest miss**. Click any other dot to switch.")
    sel = s.loc[idx]
    st.markdown(f"**{len(sel)} selected** · average miss **{sel['abs_err'].mean():.2f} pts** · "
                f"over-predicted {int((sel['error'] < 0).sum())}, under-predicted {int((sel['error'] > 0).sum())}")

    a, b, c = st.columns(3)
    cnt = sel.groupby("region_short").size().sort_values()
    fig = go.Figure(go.Bar(x=cnt.values, y=cnt.index, orientation="h", marker_color=C.VERMIL, hovertemplate="%{y}: %{x} selected<extra></extra>"))
    a.plotly_chart(C.style(fig, "Which regions?", 380, legend=False), width="stretch")
    full = C.scored(); keys = set(zip(sel["region_name"], sel["month_start"]))
    rows = full[[(r, m) in keys for r, m in zip(full["region_name"], full["month_start"])]]
    imp = rows[[f"shap_{f}" for f in N.FEATURES]].abs().mean(); imp.index = [N.LABELS[f[5:]] for f in imp.index]; imp = imp.sort_values().tail(8)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=C.BLUE, hovertemplate="%{y}: %{x:.2f} pts<extra></extra>"))
    b.plotly_chart(C.style(fig, "What drove those predictions?", 380, legend=False), width="stretch")
    reg_pick = list(dict.fromkeys(sel["region_name"]))[:4]
    tl = C.q("Timeline of selected regions", """SELECT region_short, month_start, yoy_all, pred_yoy, target_yoy FROM nowcast_scored WHERE region_name IN (SELECT unnest(?::VARCHAR[])) AND month_start >= CAST('2024-01-01' AS DATE) ORDER BY 2""", reg_pick) if False else \
        full[full["region_name"].isin(reg_pick) & (full["month_start"] >= "2024-01-01")]
    fig = go.Figure(); C.target_band(fig, False)
    for i, (rg, t) in enumerate(tl.groupby("region_short")):
        col = [C.BLUE, C.ORANGE, C.GREEN, C.PINK][i % 4]
        fig.add_scatter(x=t["month_start"], y=t["yoy_all"], mode="lines", line=dict(color=col, width=2), name=rg, hovertemplate=rg + " %{x|%b %Y}: %{y:.1f}%<extra></extra>")
    for _, r in sel[sel["region_name"].isin(reg_pick)].iterrows():
        fig.add_scatter(x=[r["month_start"] + pd.DateOffset(months=1)] * 2, y=[r["target_yoy"], r["pred_yoy"]], mode="lines+markers", line=dict(color=C.BLACK, width=3),
                        marker=dict(size=[9, 9], symbol=["circle", "x"]), showlegend=False, hovertemplate="actual vs predicted<extra></extra>")
    c.plotly_chart(C.style(fig, "Their path (● actual, × predicted)", 380), width="stretch")
    st.dataframe(sel.assign(month=sel["month_start"].dt.strftime("%b %Y"))[["region_short", "month", "yoy_all", "pred_yoy", "target_yoy", "error", "p_breach"]]
                 .rename(columns={"region_short": "Region", "month": "Month", "yoy_all": "Inflation then", "pred_yoy": "Predicted next", "target_yoy": "Actual next", "error": "Miss", "p_breach": "Risk >4%"})
                 .round(2).sort_values("Miss", key=abs, ascending=False), width="stretch", hide_index=True)
