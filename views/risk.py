"""Risk model: will a region's inflation breach 4% next month? ROC / PR / Lift with threshold + cost sliders,
SHAP + PDP explanations, decision-boundary projection, and a what-if scenario engine."""
from __future__ import annotations
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src import common as C, nowcast as N

LAB = N.LABELS


@st.cache_resource
def _pca():
    s = C.scored()
    sc = StandardScaler().fit(s[N.FEATURES]); pca = PCA(2, random_state=0).fit(sc.transform(s[N.FEATURES]))
    return sc, pca


def _curves(y, p, thr, cfn, cfp):
    st.subheader("1 · How good is the early warning? (out-of-sample months only)")
    grid = np.linspace(0, 1, 101)
    cost = [cfn * ((p < t) & (y == 1)).sum() + cfp * ((p >= t) & (y == 0)).sum() for t in grid]
    best = grid[int(np.argmin(cost))]
    pred = p >= thr
    tp, fp, fn, tn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    cur_cost = cfn * fn + cfp * fp
    cf_pred = None
    k = st.columns(5)
    k[0].metric("Catches real breaches (recall)", f"{100 * tp / max(tp + fn, 1):.0f}%")
    k[1].metric("Alarms that were right (precision)", f"{100 * tp / max(tp + fp, 1):.0f}%")
    k[2].metric("Missed breaches", fn); k[3].metric("False alarms", fp)
    k[4].metric("Total cost at this cutoff", f"{cur_cost:,.0f}", f"best cutoff {best:.2f} → {min(cost):,.0f}", delta_color="off")
    a, b, c = st.columns(3)
    fpr, tpr, _ = roc_curve(y, p)
    fig = go.Figure(); fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=C.BLUE, width=3), name="Model", hovertemplate="False alarm rate %{x:.2f}<br>Catch rate %{y:.2f}<extra></extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999", dash="dot"), name="Coin flip")
    fig.add_scatter(x=[fp / max(fp + tn, 1)], y=[tp / max(tp + fn, 1)], mode="markers", marker=dict(size=14, color=C.VERMIL, symbol="diamond"), name=f"cutoff {thr:.2f}")
    fig.update_xaxes(title="False-alarm rate"); fig.update_yaxes(title="Catch rate")
    a.plotly_chart(C.style(fig, f"ROC · AUC {roc_auc_score(y, p):.3f}", 340), width="stretch")
    pr, rc, _ = precision_recall_curve(y, p)
    fig = go.Figure(); fig.add_scatter(x=rc, y=pr, mode="lines", line=dict(color=C.BLUE, width=3), name="Model", hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>")
    fig.add_hline(y=y.mean(), line_dash="dot", line_color="#999", annotation_text=f"base rate {y.mean():.2f}")
    fig.add_scatter(x=[tp / max(tp + fn, 1)], y=[tp / max(tp + fp, 1)], mode="markers", marker=dict(size=14, color=C.VERMIL, symbol="diamond"), name=f"cutoff {thr:.2f}")
    fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision", range=[0, 1.02])
    b.plotly_chart(C.style(fig, f"Precision–Recall · AP {average_precision_score(y, p):.3f}", 340), width="stretch")
    o = np.argsort(-p); yy = y[o]; dec = np.array_split(yy, 10)
    lift = [d.mean() / max(y.mean(), 1e-9) for d in dec]; gain = np.cumsum([d.sum() for d in dec]) / max(y.sum(), 1)
    fig = go.Figure(go.Bar(x=[f"{i + 1}" for i in range(10)], y=lift, marker_color=C.SKY, name="Lift", hovertemplate="Decile %{x}: %{y:.2f}× the average<extra></extra>"))
    fig.add_scatter(x=[f"{i + 1}" for i in range(10)], y=gain, mode="lines+markers", yaxis="y2", line=dict(color=C.VERMIL, width=2.5), name="Cumulative gain",
                    hovertemplate="Top %{x}0%: %{y:.0%} of breaches found<extra></extra>")
    fig.add_hline(y=1, line_dash="dot", line_color="#999")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], tickformat=".0%", title="Gain"))
    fig.update_xaxes(title="Decile of risk score (1 = riskiest)"); fig.update_yaxes(title="Lift")
    c.plotly_chart(C.style(fig, "Lift & gain", 340), width="stretch")
    d, e = st.columns([1.4, 1])
    fig = go.Figure(); fig.add_scatter(x=grid, y=cost, mode="lines", line=dict(color=C.BLUE, width=3), name="Total cost", hovertemplate="Cutoff %{x:.2f}: cost %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=[best], y=[min(cost)], mode="markers", marker=dict(size=14, color=C.GREEN, symbol="star"), name="Cheapest cutoff")
    fig.add_scatter(x=[thr], y=[cur_cost], mode="markers", marker=dict(size=14, color=C.VERMIL, symbol="diamond"), name="Your cutoff")
    fig.update_xaxes(title="Alarm cutoff (raise alarm when risk ≥ cutoff)"); fig.update_yaxes(title="Cost units")
    d.plotly_chart(C.style(fig, "Business-cost curve (change the cost sliders)", 320), width="stretch")
    cm = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(go.Heatmap(z=cm, x=["No alarm", "Alarm"], y=["Stayed ≤4%", "Went >4%"], text=cm, texttemplate="%{text}", colorscale=C.SEQUENTIAL, showscale=False,
                               hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    e.plotly_chart(C.style(fig, "What happened at your cutoff", 320, legend=False), width="stretch")


def _explain(te: pd.DataFrame):
    st.subheader("2 · Why does the model say that? (SHAP & partial dependence)")
    sh = te[[f"shap_{f}" for f in N.FEATURES]].copy(); sh.columns = N.FEATURES
    imp = sh.abs().mean().sort_values()
    a, b = st.columns(2)
    fig = go.Figure(go.Bar(x=imp.values, y=[LAB[f] for f in imp.index], orientation="h", marker_color=C.BLUE, hovertemplate="%{y}: average push %{x:.2f} pts<extra></extra>"))
    a.plotly_chart(C.style(fig, "Global importance: average push on next month's change", 440, legend=False), width="stretch")
    top = list(imp.index[-8:][::-1]); samp = te.sample(min(100, len(te)), random_state=1)
    fig = go.Figure(); rng = np.random.default_rng(0)
    for i, f in enumerate(top):
        v = samp[f]; norm_v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
        fig.add_scatter(x=samp[f"shap_{f}"], y=i + rng.uniform(-0.28, 0.28, len(samp)), mode="markers", showlegend=False,
                        marker=dict(size=6, color=norm_v, colorscale=C.SEQUENTIAL, cmin=0, cmax=1, showscale=(i == 0), colorbar=dict(title="Feature<br>value", tickvals=[0, 1], ticktext=["low", "high"])),
                        customdata=np.c_[samp["region_short"], v], hovertemplate="%{customdata[0]}<br>value %{customdata[1]:.1f} → push %{x:+.2f} pts<extra></extra>")
    fig.update_yaxes(tickvals=list(range(len(top))), ticktext=[LAB[f] for f in top], autorange="reversed"); fig.add_vline(x=0, line_color="#999")
    fig.update_xaxes(title="← pulls inflation down · pushes it up →")
    b.plotly_chart(C.style(fig, "Beeswarm: each dot is a region-month", 440, legend=False), width="stretch")

    c, d = st.columns(2)
    s = C.scored(); regs = [r for r in C.region_names() if r != "Philippines"]
    reg0 = st.session_state["region"] if st.session_state["region"] in regs else regs[0]
    region = c.selectbox("Explain this region…", regs, index=regs.index(reg0), key="rk_region")
    rows = s[s["region_name"] == region].dropna(subset=["shap_base"]); ms = list(rows["month_start"])
    month = c.select_slider("…in this month", options=ms, value=ms[-1], format_func=C.fmt_month, key="rk_month")
    r = rows[rows["month_start"] == month].iloc[0]
    contrib = pd.Series({f: r[f"shap_{f}"] for f in N.FEATURES}); order = contrib.abs().sort_values(ascending=False)
    keep = list(order.index[:7]); other = contrib.drop(keep).sum()
    fig = go.Figure(go.Waterfall(x=["Typical change"] + [f"{LAB[f]} = {r[f]:.1f}" for f in keep] + ["Everything else", "Predicted change"],
                                 y=[r["shap_base"]] + [contrib[f] for f in keep] + [other, 0], measure=["absolute"] + ["relative"] * (len(keep) + 1) + ["total"],
                                 increasing=dict(marker_color=C.VERMIL), decreasing=dict(marker_color=C.BLUE), totals=dict(marker_color=C.BLACK),
                                 hovertemplate="%{x}: %{y:+.2f} pts<extra></extra>"))
    fig.update_xaxes(tickangle=-35); fig.update_yaxes(title="change in yoy (pts)")
    c.plotly_chart(C.style(fig, f"Waterfall: {region.split(' (')[0]}, {C.fmt_month(month)} → predicted yoy {r['pred_yoy']:.1f}%", 430, legend=False), width="stretch")
    feat = d.selectbox("Partial dependence for…", N.FEATURES, index=N.FEATURES.index("yoy_all"), format_func=lambda f: LAB[f])
    reg = C.models()["reg"]; grid = np.linspace(te[feat].quantile(0.05), te[feat].quantile(0.95), 25); ice_rows = te.sample(30, random_state=2)
    ice = np.array([reg.predict(ice_rows[N.FEATURES].assign(**{feat: g})) for g in grid])      # (25, 30)
    fig = go.Figure()
    for j in range(ice.shape[1]):
        fig.add_scatter(x=grid, y=ice[:, j], mode="lines", line=dict(color="rgba(86,180,233,0.35)", width=1), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=grid, y=ice.mean(axis=1), mode="lines", line=dict(color=C.VERMIL, width=4), name="Average (PDP)", hovertemplate=LAB[feat] + " = %{x:.1f}: %{y:+.2f} pts<extra></extra>")
    fig.update_xaxes(title=LAB[feat]); fig.update_yaxes(title="predicted change in yoy (pts)")
    d.plotly_chart(C.style(fig, "PDP (thick) over ICE lines (thin)", 430), width="stretch")


def _boundary(te: pd.DataFrame, thr: float):
    st.subheader("3 · Where does the model draw the line? (decision-boundary projection)")
    st.caption("All 15 signals are squeezed into 2 axes (PCA) so we can see them. Background = the model's risk if a region sat at that spot; × = months the model got wrong.")
    mode = st.radio("Background", ["The model itself (projected)", "k-nearest neighbours"], horizontal=True)
    kk = st.slider("k (neighbours)", 3, 41, 15, step=2, disabled=mode.startswith("The model"))
    sc, pca = _pca(); s = C.scored()
    xs = np.linspace(s["pc1"].min() - 0.5, s["pc1"].max() + 0.5, 70); ys = np.linspace(s["pc2"].min() - 0.5, s["pc2"].max() + 0.5, 70)
    gx, gy = np.meshgrid(xs, ys); G = np.c_[gx.ravel(), gy.ravel()]
    if mode.startswith("The model"):
        feats = pd.DataFrame(sc.inverse_transform(pca.inverse_transform(G)), columns=N.FEATURES)
        z = C.models()["clf"].predict_proba(feats)[:, 1]
    else:
        lab = s.dropna(subset=["breach_actual"]); z = KNeighborsClassifier(kk).fit(lab[["pc1", "pc2"]], lab["breach_actual"].astype(int)).predict_proba(G)[:, 1]
    fig = go.Figure(go.Contour(x=xs, y=ys, z=z.reshape(gx.shape), colorscale=C.DIVERGING, zmin=0, zmax=1, contours=dict(start=0, end=1, size=0.1), opacity=0.55,
                               colorbar=dict(title="Risk of >4%"), hoverinfo="skip", line_width=0))
    fig.add_contour(x=xs, y=ys, z=z.reshape(gx.shape), contours=dict(start=thr, end=thr, coloring="none"), line=dict(color="black", width=3), showscale=False, hoverinfo="skip", name="cutoff")
    wrong = ((te["p_breach"] >= thr) != (te["breach_actual"] == 1))
    for flag, nm, col in [(1, "Went above 4%", C.VERMIL), (0, "Stayed ≤4%", C.BLUE)]:
        t = te[te["breach_actual"] == flag]
        fig.add_scatter(x=t["pc1"], y=t["pc2"], mode="markers", name=nm, marker=dict(size=7, color=col, line=dict(color="white", width=0.8)),
                        customdata=np.c_[t["region_short"], t["month_start"].dt.strftime("%b %Y"), t["p_breach"]], hovertemplate="%{customdata[0]} · %{customdata[1]}<br>risk %{customdata[2]:.0%}<extra></extra>")
    w = te[wrong]
    fig.add_scatter(x=w["pc1"], y=w["pc2"], mode="markers", name="Model wrong", marker=dict(size=13, color="rgba(0,0,0,0)", symbol="x", line=dict(color="black", width=2)),
                    customdata=np.c_[w["region_short"], w["month_start"].dt.strftime("%b %Y"), w["p_breach"]], hovertemplate="WRONG: %{customdata[0]} · %{customdata[1]}<br>risk %{customdata[2]:.0%}<extra></extra>")
    fig.update_xaxes(title=f"PC1 (overall price pressure)"); fig.update_yaxes(title="PC2")
    st.plotly_chart(C.style(fig, f"Projection explains {100 * pca.explained_variance_ratio_.sum():.0f}% of the signal variation · {len(w)} of {len(te)} months misclassified at cutoff {thr:.2f}", 520), width="stretch")


def _whatif():
    st.subheader("4 · What-if engine: change the inputs, watch the risk move")
    s = C.scored(); regs = [r for r in C.region_names() if r != "Philippines"]
    reg0 = st.session_state["region"] if st.session_state["region"] in regs else regs[0]
    a, b, c = st.columns(3)
    region = a.selectbox("Region", regs, index=regs.index(reg0), key="wi_region")
    grp = b.selectbox("Price shock in…", ["01", "07", "04"], format_func=lambda x: C.SHORT[x], key="wi_grp")
    shock = c.slider("Shock size (%)", -10, 25, 5, key="wi_shock")
    extra = st.slider("Extra across-the-board price jump this month (%)", -1.0, 3.0, 0.0, 0.1)
    with st.expander("Hyperparameters of the model (what-if on the model itself)"):
        h1, h2, h3 = st.columns(3)
        ne = h1.slider("Trees", 50, 600, 300, 50); md = h2.slider("Tree depth", 1, 5, 3); lr = h3.slider("Learning rate", 0.01, 0.2, 0.05, 0.01)
        use = st.checkbox("Use these settings for the scenario", False)
        reg_t, clf_t = C.tuned_models(ne, md, lr); te = s[s["is_test"]]
        base_reg, base_clf = C.models()["reg"], C.models()["clf"]
        mae_t = float((reg_t.predict(te[N.FEATURES]) + te["yoy_all"] - te["target_yoy"]).abs().mean())
        auc_t = roc_auc_score(te["breach_actual"].astype(int), clf_t.predict_proba(te[N.FEATURES])[:, 1])
        m = C.metrics()
        x1, x2 = st.columns(2)
        x1.metric("Typical miss (MAE, pts)", f"{mae_t:.3f}", f"{mae_t - m['reg_mae_model']:+.3f} vs default", delta_color="inverse")
        x2.metric("Breach AUC", f"{auc_t:.3f}", f"{auc_t - m['clf_auc']:+.3f} vs default")
    reg, clf = (reg_t, clf_t) if use else (C.models()["reg"], C.models()["clf"])
    row = s[(s["region_name"] == region) & s["is_latest"]].iloc[[0]].copy()
    w = C.q("Group weight in region", """SELECT w.weight_in_region AS w FROM fact_weight w JOIN dim_region r USING (region_id) JOIN dim_item i USING (item_id)
        WHERE r.region_name = ? AND i.item_code = ?""", region, grp)["w"].iloc[0]
    sc = row.copy(); gcol = C.FEATURE_GROUP[grp]; mcol = gcol.replace("yoy_", "mom_")
    delta = w / 100 * shock + extra
    sc[gcol] += shock; sc[mcol] += shock
    for col in ["yoy_all", "mom_all", "mom3_all", "gap_vs_nat"]:
        sc[col] += delta
    b0, b1 = N.predict(reg, clf, row).iloc[0], N.predict(reg, clf, sc).iloc[0]
    k = st.columns(4)
    k[0].metric("Latest month", C.fmt_month(row["month_start"].iloc[0])); k[1].metric("Inflation now → after shock", f"{b1['yoy_all']:.1f}%", f"{delta:+.2f} pts", delta_color="inverse")
    k[2].metric("Expected next month", f"{b1['pred_yoy']:.1f}%", f"{b1['pred_yoy'] - b0['pred_yoy']:+.2f} vs no shock", delta_color="inverse")
    k[3].metric("Chance above 4%", f"{100 * b1['p_breach']:.0f}%", f"{100 * (b1['p_breach'] - b0['p_breach']):+.0f} pts", delta_color="inverse")
    fig = go.Figure(go.Bar(x=["No shock", "With your scenario"], y=[100 * b0["p_breach"], 100 * b1["p_breach"]], marker_color=[C.BLUE, C.VERMIL],
                           text=[f"{100 * b0['p_breach']:.0f}%", f"{100 * b1['p_breach']:.0f}%"], hovertemplate="%{x}: %{y:.0f}% chance<extra></extra>"))
    fig.update_yaxes(range=[0, 105], title="Chance of >4% next month (%)")
    st.plotly_chart(C.style(fig, f"{region.split(' (')[0]}: risk before vs after", 300, legend=False), width="stretch")
    C.sql_box("Group weight in region")


def render():
    st.header("Risk model — will a region break the 4% ceiling next month?")
    s = C.scored(); m = C.metrics(); te = s[s["is_test"]].reset_index(drop=True)
    st.caption(f"Trained on months before {C.fmt_month(m['test_start'])}; every score below uses only the {len(te)} later region-months the model never saw.")
    only = st.checkbox("Only the selected region", False) and st.session_state["region"] != "Philippines"
    sub = te[te["region_name"] == st.session_state["region"]] if only else te
    y = sub["breach_actual"].astype(int).values
    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Alarm cutoff", 0.05, 0.95, 0.5, 0.05); cfn = c2.slider("Cost of a MISSED breach", 1, 20, 5); cfp = c3.slider("Cost of a FALSE alarm", 1, 20, 1)
    if len(np.unique(y)) < 2:
        st.warning("This selection has only one outcome type, so curves can't be drawn. Untick the region filter.")
    else:
        _curves(y, sub["p_breach"].values, thr, cfn, cfp)
    _explain(te)
    _boundary(te, thr)
    _whatif()
