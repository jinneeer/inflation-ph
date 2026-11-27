"""Train + evaluate everything once; write small artifacts the app loads.   python -m src.train"""
from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, shap
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from src import forecast as F, nowcast as N
from src.db import connect

ART = Path(__file__).resolve().parents[1] / "data" / "artifacts"
FORECAST_ORIGINS = ["2023-10-01", "2024-10-01", "2025-10-01", "2026-04-01"]
NOWCAST_ORIGINS = ["2023-10-01", "2024-10-01", "2025-10-01"]


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    con = connect()
    # ---- national forecast backtest -------------------------------------------------------------
    cpi = F.national_cpi(con)
    fe = F.rolling_origin(cpi, FORECAST_ORIGINS, horizon=12)
    fe.to_parquet(ART / "forecast_backtest.parquet")
    print(fe.groupby("model")[["abs_err"]].mean().round(2).T.to_string())
    print("80% interval coverage:", fe.groupby("model")["in80"].mean().dropna().round(2).to_dict())

    # ---- regional nowcast -----------------------------------------------------------------------
    panel = N.load_panel(con)
    d = N.usable(panel)
    train = d[(d["month_start"] < N.TEST_START) & d["target_yoy"].notna()]
    reg, clf = N.fit_models(train)
    scored = N.predict(reg, clf, d)                       # every row; `is_test` marks honest out-of-sample rows
    scored["is_test"] = (scored["month_start"] >= N.TEST_START) & scored["target_yoy"].notna()
    scored["is_latest"] = scored["target_yoy"].isna()
    scored["breach_actual"] = np.where(scored["target_yoy"].notna(), (scored["target_yoy"] > N.THRESHOLD).astype(float), np.nan)
    scored["error"] = scored["target_yoy"] - scored["pred_yoy"]

    # SHAP on the regressor (change in yoy) for every row
    ex = shap.TreeExplainer(reg)
    sv = ex.shap_values(scored[N.FEATURES])
    for i, f in enumerate(N.FEATURES):
        scored[f"shap_{f}"] = sv[:, i]
    scored["shap_base"] = float(np.ravel(ex.expected_value)[0])

    # PCA projection of the feature space (for the decision-boundary view)
    z = StandardScaler().fit_transform(scored[N.FEATURES])
    pca = PCA(n_components=2, random_state=0).fit(z)
    xy = pca.transform(z)
    scored["pc1"], scored["pc2"] = xy[:, 0], xy[:, 1]
    scored["pca_var"] = float(pca.explained_variance_ratio_.sum())
    scored.to_parquet(ART / "nowcast_scored.parquet")

    te = scored[scored["is_test"]]
    y = te["breach_actual"].astype(int)
    m = {
        "n_train": int(len(train)), "n_test": int(len(te)), "threshold": N.THRESHOLD,
        "test_start": str(N.TEST_START.date()), "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
        "reg_mae_model": float((te["pred_yoy"] - te["target_yoy"]).abs().mean()),
        "reg_mae_carry_forward": float((te["yoy_all"] - te["target_yoy"]).abs().mean()),
        "clf_auc": float(roc_auc_score(y, te["p_breach"])) if y.nunique() > 1 else None,
        "clf_ap": float(average_precision_score(y, te["p_breach"])) if y.nunique() > 1 else None,
        "clf_brier": float(brier_score_loss(y, te["p_breach"])),
        "test_positive_rate": float(y.mean()),
        "clf_accuracy_carry_forward": float(((te["yoy_all"] > N.THRESHOLD).astype(int) == y).mean()),
        "clf_accuracy_model": float(((te["p_breach"] > 0.5).astype(int) == y).mean()),
        "latest_month": str(scored["month_start"].max().date()),
    }
    bt = N.backtest(panel, NOWCAST_ORIGINS)
    bt.to_parquet(ART / "nowcast_backtest.parquet")
    m["backtest"] = bt.assign(origin=bt["origin"].dt.strftime("%Y-%m")).to_dict("records")
    (ART / "metrics.json").write_text(json.dumps(m, indent=1))
    joblib.dump({"reg": reg, "clf": clf, "features": N.FEATURES}, ART / "nowcast_models.joblib", compress=3)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "backtest"}, indent=1))
    print(bt.round(3).to_string())


if __name__ == "__main__":
    main()
