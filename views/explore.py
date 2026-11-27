"""Explore: 5 basic + 5 advanced visuals, each with tooltips, statistical annotations and an SQL preview."""
from __future__ import annotations
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from scipy import stats
from src import common as C

BASE = """FROM fact_cpi f JOIN dim_region r USING (region_id) JOIN dim_item i USING (item_id) JOIN dim_date d USING (date_id)"""


def render():
    region, (p0, p1) = st.session_state["region"], st.session_state["period"]
    d0, d1 = p0.strftime("%Y-%m-%d"), p1.strftime("%Y-%m-%d")
    st.header("Explore the numbers")
    st.caption(f"Region: **{region}** · period {C.fmt_month(p0)} – {C.fmt_month(p1)} (change in the sidebar). "
               f"Snapshot charts use the last month of the period: **{C.fmt_month(p1)}**.")
    tb, ta = st.tabs(["Basic views (5)", "Advanced views (5)"])

    with tb:
        c1, c2 = st.columns(2)
        # B1 line: region vs national + trend annotation
        s = C.q("B1 Region vs national inflation", f"""SELECT d.month_start, r.region_name, f.yoy_pct AS yoy {BASE}
            WHERE i.is_headline AND r.region_name IN (?, 'Philippines') AND d.month_start BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            AND f.yoy_pct IS NOT NULL ORDER BY 1""", region, d0, d1)
        fig = go.Figure(); C.target_band(fig)
        for name, col in [("Philippines", C.BLACK), (region, C.BLUE)]:
            t = s[s["region_name"] == name]
            if t.empty or (name == region and region == "Philippines" and col == C.BLUE):
                continue
            fig.add_scatter(x=t["month_start"], y=t["yoy"], name=name.split(" (")[0], mode="lines", line=dict(color=col, width=2.5),
                            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>" + name.split(" (")[0] + "</extra>")
        t = s[s["region_name"] == region]
        if len(t) > 3:
            slope = np.polyfit(np.arange(len(t)), t["yoy"], 1)[0] * 12
            fig.add_annotation(text=f"Trend: {slope:+.1f} pts/yr · mean {t['yoy'].mean():.1f}% · max {t['yoy'].max():.1f}%",
                               xref="paper", yref="paper", x=0.01, y=0.98, showarrow=False, bgcolor="white", font_size=11)
        fig.update_yaxes(ticksuffix="%")
        c1.plotly_chart(C.style(fig, "1 · Inflation over time", 340), width="stretch"); 
        with c1: C.sql_box("B1 Region vs national inflation")

        # B2 bar: regions at snapshot
        b = C.q("B2 Regions at snapshot month", f"""SELECT r.region_short, f.yoy_pct AS yoy {BASE}
            WHERE i.is_headline AND NOT r.is_national AND d.month_start = CAST(? AS DATE) ORDER BY 2""", d1)
        fig = go.Figure(go.Bar(x=b["yoy"], y=b["region_short"], orientation="h",
                               marker_color=[C.VERMIL if v > C.TARGET_HI else C.BLUE for v in b["yoy"]],
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.add_vline(x=C.TARGET_HI, line_dash="dash", annotation_text="4% ceiling")
        fig.add_vline(x=b["yoy"].median(), line_color=C.ORANGE, annotation_text=f"median {b['yoy'].median():.1f}%")
        c2.plotly_chart(C.style(fig, f"2 · Regions, {C.fmt_month(p1)}", 340, legend=False), width="stretch")
        with c2: C.sql_box("B2 Regions at snapshot month")

        c3, c4 = st.columns(2)
        # B3 weights
        w = C.q("B3 Basket weights", """SELECT i.item_code, i.item_name, w.weight_national AS weight FROM fact_weight w JOIN dim_region r USING (region_id)
            JOIN dim_item i USING (item_id) WHERE r.is_national AND NOT i.is_headline ORDER BY 3""")
        w["label"] = w["item_code"].map(C.SHORT)
        fig = go.Figure(go.Bar(x=w["weight"], y=w["label"], orientation="h", marker_color=C.SKY, hovertemplate="%{y}: %{x:.1f}% of spending<extra></extra>"))
        fig.update_xaxes(ticksuffix="%")
        c3.plotly_chart(C.style(fig, "3 · What households spend on (basket weights)", 360, legend=False), width="stretch")
        with c3: C.sql_box("B3 Basket weights")

        # B4 histogram
        h = C.q("B4 Distribution of group inflation", f"""SELECT f.yoy_pct AS yoy {BASE} WHERE NOT i.is_headline AND NOT r.is_national
            AND d.month_start = CAST(? AS DATE)""", d1)
        fig = go.Figure(go.Histogram(x=h["yoy"], nbinsx=20, marker_color=C.BLUE, hovertemplate="%{x}%: %{y} items<extra></extra>"))
        fig.add_vline(x=h["yoy"].mean(), line_color=C.VERMIL, annotation_text=f"mean {h['yoy'].mean():.1f}%")
        fig.add_vline(x=h["yoy"].median(), line_color=C.ORANGE, line_dash="dash", annotation_text=f"median {h['yoy'].median():.1f}%")
        fig.add_annotation(text=f"sd {h['yoy'].std():.1f} · skew {stats.skew(h['yoy']):+.1f} · n={len(h)}", xref="paper", yref="paper", x=0.99, y=0.97, showarrow=False, xanchor="right", yanchor="top")
        c4.plotly_chart(C.style(fig, f"4 · How spread out are price changes? ({C.fmt_month(p1)})", 360, legend=False), width="stretch")
        with c4: C.sql_box("B4 Distribution of group inflation")

        # B5 scatter food vs transport
        sc = C.q("B5 Food vs transport inflation", f"""SELECT r.region_short, max(CASE WHEN i.item_code='01' THEN f.yoy_pct END) AS food,
            max(CASE WHEN i.item_code='07' THEN f.yoy_pct END) AS transport, max(CASE WHEN i.is_headline THEN w.weight_national END) AS weight
            {BASE} JOIN fact_weight w USING (region_id, item_id) WHERE NOT r.is_national AND d.month_start = CAST(? AS DATE) GROUP BY 1""", d1) \
            if False else C.q("B5 Food vs transport inflation", f"""SELECT r.region_short,
            max(CASE WHEN i.item_code='01' THEN f.yoy_pct END) AS food, max(CASE WHEN i.item_code='07' THEN f.yoy_pct END) AS transport
            {BASE} WHERE NOT r.is_national AND d.month_start = CAST(? AS DATE) GROUP BY 1""", d1)
        r_, p_ = stats.pearsonr(sc["food"], sc["transport"])
        m, b0 = np.polyfit(sc["food"], sc["transport"], 1); xs = np.linspace(sc["food"].min(), sc["food"].max(), 2)
        # Name only the stand-out regions (furthest from the fit, plus the extremes) so labels never pile up
        # on narrow screens; every point still shows its name on hover.
        resid = (sc["transport"] - (m * sc["food"] + b0)).abs()
        keep = set(resid.nlargest(5).index) | {sc["food"].idxmin(), sc["food"].idxmax(), sc["transport"].idxmin(), sc["transport"].idxmax()}
        labels = [n if i in keep else "" for i, n in zip(sc.index, sc["region_short"])]
        fig = go.Figure(go.Scatter(x=sc["food"], y=sc["transport"], mode="markers+text", text=labels, customdata=sc["region_short"],
                                   textposition=C.label_positions(sc["food"], sc["transport"]), textfont_size=11,
                                   marker=dict(size=11, color=C.BLUE), hovertemplate="%{customdata}<br>Food %{x:.1f}% · Transport %{y:.1f}%<extra></extra>"))
        fig.add_scatter(x=xs, y=m * xs + b0, mode="lines", line=dict(color=C.VERMIL, dash="dash"), name="fit", hoverinfo="skip")
        fig.update_xaxes(title="Food inflation %"); fig.update_yaxes(title="Transport inflation %")
        st.plotly_chart(C.style(fig, f"5 · Do food and transport prices move together? ({C.fmt_month(p1)}) · r = {r_:.2f}, p = {p_:.3f} · hover a dot for its region", 420, legend=False), width="stretch")
        C.sql_box("B5 Food vs transport inflation")

    with ta:
        # A1 heatmap region x month (<=36 months)
        lo = max(p0, p1 - pd.DateOffset(months=35)).strftime("%Y-%m-%d")
        hm = C.q("A1 Region × month heatmap", f"""SELECT r.region_short, d.month_start, f.yoy_pct AS yoy {BASE}
            WHERE i.is_headline AND NOT r.is_national AND d.month_start BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)""", lo, d1)
        pv = hm.pivot(index="region_short", columns="month_start", values="yoy")
        pv = pv.loc[pv.iloc[:, -1].sort_values().index]
        fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=C.DIVERGING, zmid=C.TARGET_HI,
                                   colorbar=dict(title="% yoy"), hovertemplate="%{y} · %{x|%b %Y}: %{z:.1f}%<extra></extra>"))
        st.plotly_chart(C.style(fig, "6 · Heatmap: which regions ran hot, and when (blue below the 4% ceiling, red above)", 460, legend=False), width="stretch")
        C.sql_box("A1 Region × month heatmap")

        a, b_ = st.columns(2)
        # A2 treemap
        g = C.q("A2 Treemap weights × inflation", f"""SELECT i.item_code, w.weight_national AS weight, f.yoy_pct AS yoy
            {BASE} JOIN fact_weight w USING (region_id, item_id) WHERE r.is_national AND NOT i.is_headline AND d.month_start = CAST(? AS DATE)""", d1)
        g["label"] = g["item_code"].map(C.SHORT)
        fig = go.Figure(go.Treemap(labels=g["label"], parents=[""] * len(g), values=g["weight"], customdata=g["yoy"],
                                   marker=dict(colors=g["yoy"], colorscale=C.DIVERGING, cmid=C.TARGET_HI, colorbar=dict(title="% yoy")),
                                   texttemplate="%{label}<br>%{customdata:+.1f}%", hovertemplate="%{label}<br>Share %{value:.1f}% · Inflation %{customdata:+.1f}%<extra></extra>"))
        a.plotly_chart(C.style(fig, "7 · Treemap: size = share of spending, colour = inflation", 420, legend=False), width="stretch")
        with a: C.sql_box("A2 Treemap weights × inflation")

        # A3 waterfall: contribution to national inflation
        g["contrib"] = g["weight"] / 100 * g["yoy"]
        head = C.q("A3 National headline", f"""SELECT f.yoy_pct AS yoy {BASE} WHERE r.is_national AND i.is_headline AND d.month_start = CAST(? AS DATE)""", d1)["yoy"].iloc[0]
        gg = g.sort_values("contrib", ascending=False)
        resid = head - gg["contrib"].sum()
        fig = go.Figure(go.Waterfall(x=[C.wrap_label(l_) for l_ in list(gg["label"]) + ["Rounding / weight drift", "National inflation"]],
                                     y=list(gg["contrib"]) + [resid, 0], measure=["relative"] * (len(gg) + 1) + ["total"],
                                     increasing=dict(marker_color=C.VERMIL), decreasing=dict(marker_color=C.BLUE), totals=dict(marker_color=C.BLACK),
                                     hovertemplate="%{x}: %{y:+.2f} pts<extra></extra>"))
        fig.update_xaxes(tickangle=-90, tickfont_size=11); fig.update_yaxes(title="percentage points")
        b_.plotly_chart(C.style(fig, f"8 · Waterfall: who contributes to national inflation ({C.fmt_month(p1)})", 420, legend=False), width="stretch")
        with b_: C.sql_box("A3 National headline")

        c, d = st.columns(2)
        # A4 correlation of groups over time, computed in SQL
        co = C.q("A4 Correlation of group inflation", f"""WITH x AS (SELECT i.item_code, d.month_start, f.yoy_pct AS yoy {BASE}
              WHERE r.region_name = ? AND NOT i.is_headline AND f.yoy_pct IS NOT NULL AND d.month_start BETWEEN CAST(? AS DATE) AND CAST(? AS DATE))
            SELECT a.item_code AS a, b.item_code AS b, corr(a.yoy, b.yoy) AS r FROM x a JOIN x b USING (month_start) GROUP BY 1, 2""", region, d0, d1)
        cm = co.pivot(index="a", columns="b", values="r"); lab = [C.SHORT[c_] for c_ in cm.index]
        fig = go.Figure(go.Heatmap(z=cm.values, x=lab, y=lab, colorscale=C.DIVERGING, zmin=-1, zmax=1, zmid=0, colorbar=dict(title="r"),
                                   hovertemplate="%{y} × %{x}: r = %{z:.2f}<extra></extra>"))
        fig.update_layout(yaxis_autorange="reversed"); fig.update_xaxes(showticklabels=False); fig.update_yaxes(tickfont_size=11)
        c.plotly_chart(C.style(fig, "9 · Which groups rise and fall together? (correlation; columns follow the same order as rows, hover for names)", 440, legend=False), width="stretch")
        with c: C.sql_box("A4 Correlation of group inflation")

        # A5 slope chart then vs now
        old = (p1 - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        sl = C.q("A5 Slope chart (a year ago vs now)", f"""SELECT r.region_short, d.month_start, f.yoy_pct AS yoy {BASE} WHERE i.is_headline AND NOT r.is_national
            AND d.month_start IN (CAST(? AS DATE), CAST(? AS DATE))""", old, d1)
        fig = go.Figure()
        for reg_, t in sl.groupby("region_short"):
            t = t.sort_values("month_start")
            if len(t) < 2: continue
            up = t["yoy"].iloc[1] > t["yoy"].iloc[0]
            fig.add_scatter(x=["A year earlier", "Now"], y=t["yoy"], mode="lines+markers", name=reg_, line=dict(color=C.VERMIL if up else C.BLUE, width=1.8),
                            hovertemplate=reg_ + ": %{y:.1f}%<extra></extra>")
        fig.add_hrect(y0=C.TARGET_LO, y1=C.TARGET_HI, fillcolor=C.GREEN, opacity=0.1, line_width=0)
        d.plotly_chart(C.style(fig, "10 · Slope chart: each line is a region (red = rose, blue = fell)", 440, legend=False), width="stretch")
        with d: C.sql_box("A5 Slope chart (a year ago vs now)")
