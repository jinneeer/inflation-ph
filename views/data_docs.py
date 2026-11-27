"""Data & SQL: ER diagram, ETL pipeline, live data-quality checks, query log with latency, model card."""
from __future__ import annotations
import pandas as pd, streamlit as st
from src import common as C
from src.db import ROOT

ER = """digraph ER { rankdir=LR; node [shape=record, fontname="Google Sans", color="#0E7490", fontcolor="#082F49", fontsize=11, style=filled, fillcolor="#E0F4FD"];
 fact_cpi [fillcolor="#BAE6FD", label="{fact_cpi|region_id (FK)\\litem_id (FK)\\ldate_id (FK)\\lcpi\\lyoy_pct\\l}"];
 fact_weight [fillcolor="#BAE6FD", label="{fact_weight|region_id (FK)\\litem_id (FK)\\lweight_national\\lweight_in_region\\l}"];
 dim_region [label="{dim_region|region_id (PK)\\lregion_name\\lregion_short\\lis_national\\l}"];
 dim_item [label="{dim_item|item_id (PK)\\litem_code\\litem_name\\lis_headline\\l}"];
 dim_date [label="{dim_date|date_id (PK)\\lmonth_start\\lyear, month, quarter\\l}"];
 nowcast_scored [fillcolor="#DCFCE7", label="{nowcast_scored (model output)|region, month, features\\lpred_yoy, p_breach\\lSHAP columns, PCA coords\\l}"];
 fact_cpi -> dim_region; fact_cpi -> dim_item; fact_cpi -> dim_date; fact_weight -> dim_region; fact_weight -> dim_item;
 nowcast_scored -> dim_region [style=dashed]; }"""
PIPE = """digraph P { rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#E0F4FD", fontname="Google Sans", color="#0E7490", fontcolor="#082F49", fontsize=11];
 a [label="PSA OpenSTAT\\n5 CSV exports\\n(CPI, yoy, weights)"]; b [label="src/etl.py\\nparse + melt\\ndetect file by content"]; c [label="sql/01_build_warehouse.sql\\nDuckDB star schema"];
 d [label="sql/02_quality_checks.sql\\n8 gates"]; e [label="Parquet warehouse\\ndata/warehouse"]; f [label="sql/03_features.sql\\nlags, m/m, next-month target"];
 g [label="src/train.py\\nSARIMA · ETS · GBM · SHAP · PCA"]; h [label="artifacts/"]; i [label="Streamlit app\\nDuckDB queries <1,000 rows"];
 a -> b -> c -> d -> e -> f -> g -> h -> i; e -> i; }"""


def render():
    st.header("Data, SQL & method")
    t1, t2, t3, t4 = st.tabs(["Schema & pipeline", "Data-quality checks", "Query log & latency", "Model card & honesty notes"])
    with t1:
        a, b = st.columns(2)
        a.subheader("ER diagram"); a.graphviz_chart(ER)
        b.subheader("ETL pipeline"); b.graphviz_chart(PIPE)
        counts = C.q("Table row counts", """SELECT 'fact_cpi' AS tbl, count(*) AS rows FROM fact_cpi UNION ALL SELECT 'fact_weight', count(*) FROM fact_weight
            UNION ALL SELECT 'dim_region', count(*) FROM dim_region UNION ALL SELECT 'dim_item', count(*) FROM dim_item UNION ALL SELECT 'dim_date', count(*) FROM dim_date
            UNION ALL SELECT 'nowcast_scored', count(*) FROM nowcast_scored""")
        st.dataframe(counts, hide_index=True, width="stretch"); C.sql_box("Table row counts")
        st.caption("Source: Philippine Statistics Authority OpenSTAT — CPI for All Income Households by Commodity Group (2018=100), year-on-year change, and 2018-based basket weights. "
                   "19 geographies × 14 commodity groups × 106 months (Jan 2018 – Oct 2026).")
    with t2:
        sql = (ROOT / "sql" / "02_quality_checks.sql").read_text()
        res = C.con().execute(sql).df(); res.columns = ["Check", "Violations"]; res["Result"] = res["Violations"].map(lambda v: "✅ pass" if v == 0 else "❌ fail")
        st.dataframe(res, hide_index=True, width="stretch")
        with st.expander("SQL of the checks"):
            st.code(sql, language="sql")
    with t3:
        log = pd.DataFrame(st.session_state.get("qlog", []))
        if not log.empty:
            st.dataframe(log[["query", "rows", "ms"]].rename(columns={"ms": "latency (ms)"}), hide_index=True, width="stretch")
            st.caption("Latency is measured per query when it first runs; repeats within a session are served from cache. Every chart receives fewer than 1,000 rows.")
            for e in st.session_state["qlog"]:
                with st.expander(e["query"]):
                    st.code(e["sql"], language="sql")
        else:
            st.info("Open the other pages to populate the log.")
    with t4:
        m = C.metrics()
        st.markdown(f"""
**What the models do.** (1) *National forecast*: SARIMA(1,1,1)(1,1,0)₁₂ and damped-trend ETS on log CPI, with 80% and 95% bands. (2) *Regional early warning*: gradient-boosted trees predict how much each region's
inflation moves next month and the probability it ends above the 4% ceiling. Trained on months before {C.fmt_month(m['test_start'])}; scored on the {m['n_test']} later region-months.

| | Model | Carry this month forward |
|---|---|---|
| Regional next-month miss (MAE, pts) | {m['reg_mae_model']:.2f} | {m['reg_mae_carry_forward']:.2f} |
| Breach call accuracy | {100 * m['clf_accuracy_model']:.1f}% | {100 * m['clf_accuracy_carry_forward']:.1f}% |
| Breach ranking quality (AUC / AP) | {m['clf_auc']:.3f} / {m['clf_ap']:.3f} | – |

**Honest reading.** Inflation is very persistent, so "next month ≈ this month" is a strong baseline, and the regional model does not beat it on average error. Its value is the *explanation* (SHAP), the *risk probability*
and the *what-if* engine, not a lower headline error. About {100 * m['test_positive_rate']:.0f}% of the out-of-sample months were above 4%, so accuracy figures are not inflated by an easy class balance.

**Limits.** The what-if on prices is a first-order basket-weight approximation. Data are group-level (14 groups), not individual products. Colour palette: Okabe-Ito and Cividis (colour-blind safe).
""")
