"""BantayPresyo — Philippine inflation dashboard (PSA OpenSTAT CPI) · Streamlit + DuckDB."""
import importlib
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BantayPresyo", page_icon=":material/shopping_cart:", layout="wide",
                   initial_sidebar_state="expanded")
from src import common as C  # noqa: E402

C.inject_css()
ASSETS = Path(__file__).parent / "assets"
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

# (title, view module, Material icon) — the sidebar menu, in order
PAGES = [("Price Check", "simple", "shopping_cart"), ("Explore", "explore", "bar_chart"),
         ("Forecast Lab", "forecast_lab", "insights"), ("Risk Model", "risk", "warning"),
         ("Where it misses", "errors", "target"), ("Data & SQL", "data_docs", "database")]
nav = st.navigation([st.Page(importlib.import_module(f"views.{mod}").render, title=title, icon=f":material/{icon}:",
                             url_path=mod, default=(i == 0)) for i, (title, mod, icon) in enumerate(PAGES)],
                    position="sidebar")
st.session_state["qlog"] = []

with st.sidebar:
    st.markdown('<p class="pc-label">Filters · apply to every page</p>', unsafe_allow_html=True)
    st.selectbox("Region", C.region_names(), key="region")
    ms = C.months()
    st.select_slider("Period", options=ms, value=(pd.Timestamp("2019-01-01"), ms[-1]), format_func=C.fmt_month, key="period")
    st.divider()
    latency = st.container()

nav.run()

log = st.session_state["qlog"]
with latency:
    if log:
        st.caption(f":material/bolt: {len(log)} SQL queries · {sum(e['ms'] for e in log):.0f} ms (cached runs are instant)")
    st.caption("Source: PSA OpenSTAT CPI (2018=100) · Jan 2018 – Oct 2026")
