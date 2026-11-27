"""Regional next-month inflation: a regressor (how far will yoy move?) and a classifier (will it breach the target ceiling?)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

FEATURES = ["yoy_all", "yoy_l1", "yoy_l2", "yoy_l3", "mom_all", "mom3_all", "yoy_food", "yoy_hous", "yoy_trans",
            "mom_food", "mom_hous", "mom_trans", "nat_yoy", "gap_vs_nat", "month_num"]
LABELS = {"yoy_all": "Inflation now", "yoy_l1": "Inflation 1 mo ago", "yoy_l2": "Inflation 2 mo ago",
          "yoy_l3": "Inflation 3 mo ago", "mom_all": "Prices this month (m/m)", "mom3_all": "Prices last 3 months",
          "yoy_food": "Food inflation", "yoy_hous": "Housing & utilities inflation", "yoy_trans": "Transport inflation",
          "mom_food": "Food m/m", "mom_hous": "Housing m/m", "mom_trans": "Transport m/m",
          "nat_yoy": "National inflation", "gap_vs_nat": "Region minus national", "month_num": "Calendar month"}
THRESHOLD = 4.0          # upper bound of the 2-4% government inflation target
TEST_START = pd.Timestamp("2025-07-01")


def load_panel(con) -> pd.DataFrame:
    df = con.execute("SELECT * FROM features WHERE NOT is_national ORDER BY region_id, month_start").df()
    df["month_start"] = pd.to_datetime(df["month_start"])
    return df


def usable(df):           # rows with full lag history
    return df.dropna(subset=FEATURES)


def fit_models(train: pd.DataFrame, thr: float = THRESHOLD, n_estimators: int = 300, max_depth: int = 3,
               learning_rate: float = 0.05, seed: int = 7):
    t = train.dropna(subset=["target_yoy"])
    kw = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=0.8, random_state=seed)
    reg = GradientBoostingRegressor(**kw).fit(t[FEATURES], t["target_yoy"] - t["yoy_all"])
    clf = GradientBoostingClassifier(**kw).fit(t[FEATURES], (t["target_yoy"] > thr).astype(int))
    return reg, clf


def predict(reg, clf, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_change"] = reg.predict(df[FEATURES])
    out["pred_yoy"] = out["yoy_all"] + out["pred_change"]
    out["p_breach"] = clf.predict_proba(df[FEATURES])[:, 1]
    return out


def backtest(panel: pd.DataFrame, origins: list[str], **kw) -> pd.DataFrame:
    """Expanding window: train on months <= origin, score the next 12 months. Compare to carry-forward."""
    rows = []
    d = usable(panel)
    for o in origins:
        o = pd.Timestamp(o)
        tr = d[d["month_start"] <= o]
        te = d[(d["month_start"] > o) & (d["month_start"] <= o + pd.DateOffset(months=12)) & d["target_yoy"].notna()]
        if te.empty:
            continue
        reg, clf = fit_models(tr, **kw)
        p = predict(reg, clf, te)
        rows.append({"origin": o, "n": len(te),
                     "mae_model": float((p["pred_yoy"] - p["target_yoy"]).abs().mean()),
                     "mae_carry_forward": float((p["yoy_all"] - p["target_yoy"]).abs().mean())})
    return pd.DataFrame(rows)
