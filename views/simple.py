"""The plain-language page: pick a region, scrub through time, slide a price what-if."""
from __future__ import annotations
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from scipy.stats import norm
from src import common as C

SERIES_SQL = """SELECT d.month_start, f.cpi, f.yoy_pct AS yoy
FROM fact_cpi f JOIN dim_region r USING (region_id) JOIN dim_item i USING (item_id) JOIN dim_date d USING (date_id)
WHERE r.region_name = ? AND i.is_headline AND d.month_start >= CAST(? AS DATE) ORDER BY 1"""
GROUPS_SQL = """SELECT i.item_code, i.item_name, f.yoy_pct AS yoy, w.weight_in_region AS weight
FROM fact_cpi f JOIN dim_region r USING (region_id) JOIN dim_item i USING (item_id)
JOIN dim_date d USING (date_id) JOIN fact_weight w USING (region_id, item_id)
WHERE r.region_name = ? AND d.month_start = CAST(? AS DATE) AND NOT i.is_headline ORDER BY f.yoy_pct DESC"""
WHATIF = {"Food & drinks": "01", "Transport": "07", "Housing, water, power & fuels": "04"}


def _outlook(region: str, month: pd.Timestamp):
    """Next-month yoy outlook, chance of exceeding 4%, and a 'typical miss' band."""
    m = C.metrics()
    if region == "Philippines":
        fc = C.national_forecast("sarima", 1, (1, 1, 1), (1, 1, 0, 12), month.strftime("%Y-%m-%d"))
        mean, lo, hi = fc["yoy"].iloc[0], fc["yoy_lo80"].iloc[0], fc["yoy_hi80"].iloc[0]
        sd = max((hi - lo) / (2 * 1.2816), 1e-6)
        return mean, 1 - norm.cdf((C.TARGET_HI - mean) / sd), (hi - lo) / 2, "national forecast model (SARIMA)"
    s = C.scored()
    row = s[(s["region_name"] == region) & (s["month_start"] == month)]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r["pred_yoy"]), float(r["p_breach"]), m["reg_mae_model"], "regional machine-learning model"


def render():
    region, start = st.session_state["region"], st.session_state["period"][0]
    allm = [m for m in C.months() if m >= pd.Timestamp("2019-01-01")]
    st.header("Are prices rising too fast?")
    st.caption("Pick a region on the left · drag the month slider · try the “what if” slider.")

    month = st.select_slider("⏱ Scrub through time", options=allm, value=allm[-1], format_func=C.fmt_month, key="s_month")
    ser = C.q("Headline price series", SERIES_SQL, region, "2019-01-01")
    g = C.q("Price change by group", GROUPS_SQL, region, month.strftime("%Y-%m-%d"))
    now = ser[ser["month_start"] == month].iloc[0]
    yoy = float(now["yoy"])

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader(f"{region.split(' (')[0]} · {C.fmt_month(month)}")
        st.markdown(f"## Prices are **{yoy:+.1f}%** higher than a year ago")
        C.verdict(yoy)
        st.write(f"Something that cost **₱1,000** in {C.fmt_month(month - pd.DateOffset(years=1))} now costs about **₱{1000 * (1 + yoy / 100):,.0f}**.")
        g2 = g.assign(contrib=g["yoy"] * g["weight"] / 100).sort_values("contrib", ascending=False)
        top = g2.iloc[0]
        st.write(f"Biggest push: **{C.SHORT.get(top['item_code'], top['item_name'])}** "
                 f"({top['yoy']:+.1f}% on the year, about {100 * top['contrib'] / max(yoy, 0.01):.0f}% of the rise)." if yoy > 0.5 else
                 "No single group is pushing prices strongly right now.")
    out = _outlook(region, month)
    with right:
        st.subheader("What next month looks like")
        if out is None:
            st.info("The outlook needs a few months of history; pick a later month.")
            nxt = None
        else:
            nxt, p, miss, who = out
            a, b = st.columns(2)
            a.metric("Expected next month", f"{nxt:.1f}%", f"{nxt - yoy:+.1f} pts vs now", delta_color="inverse")
            b.metric("Chance it is above 4%", f"{100 * p:.0f}%")
            st.caption(f"From the {who}. Typical miss: about ±{miss:.1f} pts.")
        st.markdown("**🎚 What if prices of one group change?**")
        c1, c2 = st.columns([1.2, 1])
        grp = c1.selectbox("Group", list(WHATIF), key="s_grp", label_visibility="collapsed")
        shock = c2.slider("Price change (%)", -10, 20, 0, key="s_shock", label_visibility="collapsed")
        w = float(g.loc[g["item_code"] == WHATIF[grp], "weight"].iloc[0])
        effect = w / 100 * shock
        st.write(f"{grp} makes up **{w:.0f}%** of a typical household's spending here, so a {shock:+d}% change moves overall inflation by about **{effect:+.1f} pts**.")
        if nxt is not None:
            st.metric("Next month with this change", f"{nxt + effect:.1f}%", f"{effect:+.1f} pts", delta_color="inverse")

    # chart 1: the story line
    sh = ser.dropna(subset=["yoy"])
    fig = go.Figure()
    C.target_band(fig)
    past, fut = sh[sh["month_start"] <= month], sh[sh["month_start"] > month]
    fig.add_scatter(x=past["month_start"], y=past["yoy"], mode="lines", line=dict(color=C.BLUE, width=3), name="Prices vs a year ago",
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=fut["month_start"], y=fut["yoy"], mode="lines", line=dict(color="#B0B0B0", width=2, dash="dot"),
                    name="Later months (hidden from the 'then')", hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=[month], y=[yoy], mode="markers", marker=dict(size=14, color=C.VERMIL, line=dict(color="white", width=2)),
                    name="Selected month", hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>")
    if out is not None:
        nm = month + pd.DateOffset(months=1)
        fig.add_scatter(x=[nm], y=[nxt], mode="markers", name="Expected next month",
                        marker=dict(size=12, color=C.ORANGE, symbol="diamond"),
                        error_y=dict(type="constant", value=out[2], color=C.ORANGE, thickness=2))
        if shock:
            fig.add_scatter(x=[nm], y=[nxt + effect], mode="markers", name="…with your what-if",
                            marker=dict(size=12, color=C.BLACK, symbol="star"))
    fig.update_yaxes(title="% change vs a year earlier", ticksuffix="%")
    st.plotly_chart(C.style(fig, "How fast prices have been rising", 360), width="stretch")
    C.sql_box("Headline price series")

    # chart 2: what is pushing prices
    gg = g.assign(label=g["item_code"].map(C.SHORT).fillna(g["item_name"])).sort_values("yoy")
    fig2 = go.Figure(go.Bar(x=gg["yoy"], y=gg["label"], orientation="h",
                            marker_color=[C.VERMIL if v > yoy else C.BLUE for v in gg["yoy"]],
                            customdata=gg["weight"], hovertemplate="%{y}: %{x:+.1f}%<br>Share of spending: %{customdata:.1f}%<extra></extra>"))
    fig2.add_vline(x=yoy, line_dash="dash", line_color=C.BLACK, annotation_text="Overall", annotation_position="top")
    fig2.update_xaxes(title="% change vs a year earlier", ticksuffix="%")
    st.plotly_chart(C.style(fig2, "What is pushing prices (orange-red = rising faster than overall)", 420, legend=False), width="stretch")
    C.sql_box("Price change by group")
