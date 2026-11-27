"""Shared helpers: cached data access, SQL logging/latency, colour-blind-safe palette, chart styling."""
from __future__ import annotations
import json, textwrap, time
import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from src import forecast as F, nowcast as N
from src.db import ROOT, connect

ART = ROOT / "data" / "artifacts"
# Brand palette (navy · teal · green · sky · ice) plus two warm accents kept for "above target / rising" signals
NAVY, TEAL, MINT, SKY_LT, ICE = "#082F49", "#0E7490", "#22C55E", "#BAE6FD", "#F0FDFF"
BLUE, GREEN, BLACK, SKY = TEAL, MINT, NAVY, "#5FAFC6"
ORANGE, VERMIL, PINK, YELLOW = "#E69F00", "#D55E00", "#8B5CF6", "#F0E442"
DIVERGING = [[0, TEAL], [0.5, ICE], [1, VERMIL]]               # teal <-> vermilion, safe for deuteranopia/protanopia
SEQUENTIAL = [[0, SKY_LT], [0.5, TEAL], [1, NAVY]]             # single-hue ramp, light to dark
FONT = "'Google Sans', 'Google Sans Text', system-ui, sans-serif"
TARGET_LO, TARGET_HI = 2.0, 4.0
SHORT = {"0": "All items", "01": "Food & drinks", "02": "Alcohol & tobacco", "03": "Clothing & shoes",
         "04": "Housing, water, power & fuels", "05": "Household goods", "06": "Health", "07": "Transport",
         "08": "Communication", "09": "Recreation", "10": "Education", "11": "Restaurants & hotels",
         "12": "Financial services", "13": "Personal care & misc."}
FEATURE_GROUP = {"01": "yoy_food", "04": "yoy_hous", "07": "yoy_trans"}


# ---------------------------------------------------------------- data access (all through DuckDB)
@st.cache_resource
def con():
    return connect()


@st.cache_data(show_spinner=False)
def _run(sql: str, params: tuple):
    t = time.perf_counter()
    df = con().execute(sql, list(params)).df()
    return df, (time.perf_counter() - t) * 1000


def _literal(sql: str, params: tuple) -> str:
    for p in params:
        sql = sql.replace("?", repr(str(p)) if not isinstance(p, (int, float)) else str(p), 1)
    return sql


def q(label: str, sql: str, *params) -> pd.DataFrame:
    """Run a SQL query; log it (for the SQL preview + latency panel). Visuals never get >= 1,000 rows."""
    df, ms = _run(sql, tuple(params))
    assert len(df) < 1000, f"{label}: {len(df)} rows - aggregate in SQL before plotting (<1,000 rows rule)"
    st.session_state.setdefault("qlog", []).append({"query": label, "rows": len(df), "ms": round(ms, 1),
                                                    "sql": _literal(sql, tuple(params)).strip()})
    return df


def sql_box(label: str):
    """Expandable preview of the SQL that produced a visual (latest run of `label`)."""
    log = [e for e in st.session_state.get("qlog", []) if e["query"] == label]
    if not log:
        return
    e = log[-1]
    with st.expander(f"SQL · {label} · {e['rows']} rows · {e['ms']:.1f} ms"):
        st.code(e["sql"], language="sql")


@st.cache_data
def months() -> list:
    return [pd.Timestamp(m) for m in con().execute("SELECT month_start FROM dim_date ORDER BY 1").df()["month_start"]]


@st.cache_data
def region_names() -> list:
    return con().execute("SELECT region_name FROM dim_region ORDER BY is_national DESC, region_name").df()["region_name"].tolist()


@st.cache_data
def scored() -> pd.DataFrame:
    return pd.read_parquet(ART / "nowcast_scored.parquet")


@st.cache_data
def metrics() -> dict:
    return json.loads((ART / "metrics.json").read_text())


@st.cache_data
def backtests():
    return pd.read_parquet(ART / "forecast_backtest.parquet"), pd.read_parquet(ART / "nowcast_backtest.parquet")


@st.cache_resource
def models():
    try:
        return joblib.load(ART / "nowcast_models.joblib")
    except Exception:                                   # e.g. different scikit-learn version on the host
        s = scored()
        reg, clf = N.fit_models(s[(s["month_start"] < N.TEST_START) & s["target_yoy"].notna()])
        return {"reg": reg, "clf": clf, "features": N.FEATURES}


@st.cache_resource
def tuned_models(n_estimators: int, max_depth: int, learning_rate: float):
    s = scored()
    return N.fit_models(s[(s["month_start"] < N.TEST_START) & s["target_yoy"].notna()],
                        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)


@st.cache_data(show_spinner="Fitting forecast model…")
def national_forecast(model: str, horizon: int, order: tuple, seasonal: tuple, upto: str) -> pd.DataFrame:
    cpi = F.national_cpi(con())
    return F.forecast(cpi.loc[:upto], horizon, model, order, seasonal)


@st.cache_data
def national_cpi() -> pd.Series:
    return F.national_cpi(con())


# ---------------------------------------------------------------- presentation
def _wrap(text: str, width: int = 44) -> list[str]:
    """Break long chart titles onto several lines so they never run past the chart edge."""
    return textwrap.wrap(text, width) if text else []


def _legend_rows(fig: go.Figure, avail_px: int = 330) -> int:
    """Rows a horizontal legend needs on a narrow (~phone-width) chart, so space is reserved for all of them."""
    names = [t.name for t in fig.data if t.name and t.showlegend is not False]
    rows, used = (1 if names else 0), 0
    for n in names:
        w = 44 + 6.4 * len(n)
        if used and used + w > avail_px:
            rows, used = rows + 1, 0
        used += w
    return rows


def style(fig: go.Figure, title: str = "", height: int = 380, legend: bool = True) -> go.Figure:
    """House chart style. Space for the title, the legend and line labels is reserved in pixels, so no text
    lands on top of other text whatever the window width."""
    grid = "rgba(14,116,144,0.12)"
    lines = _wrap(title)
    # Labels of vertical marker lines (median, target, ...) go in a strip above the plot instead of over the data;
    # neighbouring labels point away from each other.
    vl = [a for a in fig.layout.annotations if str(a.yref).endswith("domain") and not str(a.xref).endswith(("domain", "paper"))]
    for i, a in enumerate(sorted(vl, key=lambda a: float(a.x) if isinstance(a.x, (int, float)) else 0)):
        a.update(y=1, yanchor="bottom", yshift=2, showarrow=False, font_size=11,
                 xanchor=("center" if len(vl) == 1 else "right" if i == 0 else "left"))
    strip = 20 if vl else 0
    title_px = 22 * len(lines) + 16 if lines else 12
    rows = _legend_rows(fig) if legend else 0
    legend_px = 22 * rows + 10 if rows else 0
    fig.update_layout(template="simple_white", height=height + legend_px + strip, autosize=True,
                      title=dict(text="<br>".join(lines), x=0, xref="container", xanchor="left", y=1, yref="container", yanchor="top",
                                 pad=dict(t=14, l=16), font=dict(size=15, color=NAVY, weight=600)),
                      margin=dict(l=16, r=20, t=title_px + strip, b=56 + legend_px if rows else 16, autoexpand=True),
                      showlegend=legend,
                      legend=dict(orientation="h", x=0, xref="container", xanchor="left", y=0, yref="container", yanchor="bottom",
                                  bgcolor="rgba(0,0,0,0)", font=dict(size=11), indentation=12),
                      font=dict(family=FONT, size=12, color=NAVY), uniformtext=dict(minsize=9, mode="hide"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=[TEAL, MINT, NAVY, SKY, ORANGE, VERMIL],
                      hoverlabel=dict(font_size=12, font_family=FONT, bgcolor="white", bordercolor=SKY_LT))
    fig.update_annotations(bgcolor="rgba(255,255,255,0.85)", borderpad=2)
    fig.update_xaxes(showgrid=False, linecolor=SKY_LT, tickcolor=SKY_LT, zerolinecolor=grid, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor="rgba(0,0,0,0)", tickcolor="rgba(0,0,0,0)", zerolinecolor=grid, automargin=True)
    return fig


def wrap_label(text: str, width: int = 16) -> str:
    """Break a long category name onto two or more short lines for axis ticks."""
    return "<br>".join(textwrap.wrap(str(text), width))


def label_positions(x, y) -> list[str]:
    """Text positions for a labelled scatter that keep neighbouring labels apart and inside the chart:
    of two nearby points, the upper one is labelled above and the lower one below."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx = (x - x.min()) / (np.ptp(x) or 1); ny = (y - y.min()) / (np.ptp(y) or 1)
    pos = []
    for i in range(len(x)):
        side = "left" if nx[i] > 0.85 else "right" if nx[i] < 0.15 else "center"
        near = (np.abs(nx - nx[i]) < 0.18) & (np.abs(ny - ny[i]) < 0.14); near[i] = False
        vert = "bottom" if (near & (ny > ny[i])).any() and not (near & (ny < ny[i])).any() else "top"
        pos.append(f"{vert} {side}")
    return pos


def target_band(fig: go.Figure, label: bool = True):
    text = dict(annotation_text="Government target 2–4%", annotation_position="top left", annotation_font_size=11) if label else {}
    fig.add_hrect(y0=TARGET_LO, y1=TARGET_HI, fillcolor=GREEN, opacity=0.12, line_width=0, **text)


def inject_css():
    """Modern, responsive card layout on top of the theme in .streamlit/config.toml."""
    st.markdown("""<style>
    .block-container { padding: 2rem clamp(1rem, 3vw, 3rem) 3rem; max-width: 1500px; }
    header[data-testid="stHeader"] { background: transparent; }

    /* headings scale with the window instead of a fixed size */
    h1, h2, h3 { letter-spacing: -0.02em; color: #082F49; overflow-wrap: anywhere; }
    [data-testid="stHeading"] h2, .stMarkdown h2 { font-size: clamp(1.35rem, 1rem + 1.4vw, 2rem) !important; line-height: 1.2; }
    [data-testid="stHeading"] h3, .stMarkdown h3 { font-size: clamp(1.1rem, .9rem + .8vw, 1.5rem) !important; line-height: 1.25; }

    /* columns wrap onto a new row when there is no room, rather than squeezing their contents */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 1rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: min(100%, 320px); }
    [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child [data-testid="stMetric"]) {
        min-width: min(100%, 170px); }

    /* cards: metrics, charts, diagrams */
    [data-testid="stMetric"], [data-testid="stPlotlyChart"], [data-testid="stGraphVizChart"] {
        background: #FFFFFF; border-radius: 16px;   /* ring via box-shadow so the card adds no size and never crops a chart */
        box-shadow: 0 0 0 1px #BAE6FD, 0 1px 2px rgba(8,47,73,.04), 0 6px 20px rgba(8,47,73,.06); }
    [data-testid="stMetric"] { padding: 1rem 1.2rem; height: 100%; }
    [data-testid="stGraphVizChart"] { padding: .75rem .9rem; }
    [data-testid="stGraphVizChart"] svg { max-width: 100%; height: auto; }
    [data-testid="stMetricLabel"] p { font-size: .78rem; font-weight: 500; text-transform: uppercase; letter-spacing: .04em;
        color: #0E7490; white-space: normal; overflow: visible; text-overflow: clip; }
    [data-testid="stMetricLabel"] div, [data-testid="stMetricValue"] div { white-space: normal; overflow: visible; text-overflow: clip; }
    [data-testid="stMetricValue"] { font-size: clamp(1.5rem, 1rem + 1.6vw, 2.2rem); line-height: 1.15; }
    [data-testid="stAlert"] { border-radius: 12px; }

    /* tabs as pills that wrap instead of scrolling off-screen */
    [role="tablist"] { gap: .4rem; flex-wrap: wrap; border-bottom: none !important; box-shadow: none !important; padding-bottom: .5rem; }
    [data-testid="stTab"] { background: #FFFFFF; border: 1px solid #BAE6FD; border-radius: 999px; padding: .35rem 1rem; height: auto; }
    [data-testid="stTab"]:hover { border-color: #0E7490; }
    [data-testid="stTab"][aria-selected="true"] { background: #0E7490; border-color: #0E7490; }
    [data-testid="stTab"][aria-selected="true"] p { color: #FFFFFF; font-weight: 600; }
    [data-testid="stTab"] .react-aria-SelectionIndicator { display: none; }

    /* sidebar menu: rounded links, green for the current page */
    [data-testid="stSidebar"] { box-shadow: 4px 0 24px rgba(8,47,73,.18); }
    [data-testid="stSidebarNav"] a { border-radius: 10px; margin: 2px 0; padding: .45rem .7rem; }
    [data-testid="stSidebarNav"] a:hover { background: rgba(186,230,253,.12); }
    [data-testid="stSidebarNav"] a[aria-current="page"] { background: #22C55E; }
    [data-testid="stSidebarNav"] a[aria-current="page"] span { color: #082F49 !important; font-weight: 600; }
    [data-testid="stSidebar"] hr, [data-testid="stSidebarNavSeparator"] { border-color: rgba(186,230,253,.18); }
    [data-testid="stSidebar"] .pc-label { font-size: .72rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: #BAE6FD; margin: .4rem 0 0; }

    /* hide / show the sidebar: labelled pill buttons, always visible */
    [data-testid="stSidebarCollapseButton"] { display: flex !important; visibility: visible !important; opacity: 1 !important; }
    [data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] {
        display: inline-flex !important; align-items: center; gap: .3rem; width: auto !important; height: 2rem;
        padding: 0 .7rem 0 .5rem !important; border-radius: 999px; font-family: inherit; font-size: .8rem; font-weight: 600; }
    [data-testid="stSidebarCollapseButton"] button { background: rgba(186,230,253,.14); border: 1px solid rgba(186,230,253,.35); color: #F0FDFF; }
    [data-testid="stSidebarCollapseButton"] button:hover { background: rgba(186,230,253,.28); }
    [data-testid="stSidebarCollapseButton"] button span { color: #F0FDFF !important; }
    [data-testid="stSidebarCollapseButton"] button::after { content: "Hide"; }
    [data-testid="stExpandSidebarButton"] { background: #0E7490 !important; border: none; color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(8,47,73,.25); }
    [data-testid="stExpandSidebarButton"]:hover { background: #082F49 !important; }
    [data-testid="stExpandSidebarButton"] span { color: #FFFFFF !important; }
    [data-testid="stExpandSidebarButton"]::after { content: "Menu"; }
    </style>""", unsafe_allow_html=True)


def fmt_month(d) -> str:
    return pd.Timestamp(d).strftime("%b %Y")


def verdict(yoy: float):
    if yoy > TARGET_HI:
        st.error(f"**Above target.** Prices are rising faster than the government's 2–4% comfort zone.")
    elif yoy >= TARGET_LO:
        st.success("**Within target.** Price rises are inside the government's 2–4% comfort zone.")
    else:
        st.info("**Below target.** Prices are rising more slowly than the government's 2–4% comfort zone.")
