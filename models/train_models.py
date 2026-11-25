"""Offline evaluation and artifact generation for the regional nowcast and national forecasts.

    python -m models.train_models

Reads  : data/warehouse/feat_region_month.parquet, data/warehouse (national CPI)
Writes : data/models/{metrics.json, nowcast_test, nowcast_latest, clf_test, roc_curve, pr_curve,
                      shap_values, feature_importance}.parquet

What it does (all numbers land in metrics.json and are shown on the dashboard):
  1. Rolling-origin evaluation of the regional nowcast over four 12-month windows, always
     refitting on data up to the origin, against naive baselines (this month's rate carried
     forward, the 3-month average, the training mean).
  2. The deployed models (delta regressor + breach classifier) on the most recent window
     (Nov 2025 - Sep 2026), with SHAP, ROC / PR curves and out-of-fold interval quantiles.
  3. A production fit on every labelled month, scoring the latest month for next-month estimates.
  4. National SARIMA / ETS / naive rolling backtest with 80% / 95% interval coverage.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from app.services import forecast as fsvc
from app.services import nowcast_model as nm
from app.services.nowcast_model import BASE_FEATURES, FEATURES, TARGET_RATE

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "models"
ORIGINS = ["2022-10-01", "2023-10-01", "2024-10-01", "2025-10-01"]   # last = primary window
PRIMARY = ORIGINS[-1]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # shrinkage of the model correction toward carry-forward


def quantiles(err_list) -> dict:
    q = np.quantile(np.concatenate(err_list), [0.025, 0.10, 0.90, 0.975])
    return {"q025": float(q[0]), "q10": float(q[1]), "q90": float(q[2]), "q975": float(q[3])}


# ----------------------------------------------------------------------------- helpers
def clean(obj):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def reg_metrics(actual, pred) -> dict:
    return {"MAE": mae(actual, pred), "RMSE": rmse(actual, pred)}


def cls_metrics(y, score, flagged) -> dict:
    y = np.asarray(y)
    out = {"n": int(len(y)), "positive_rate": float(y.mean()),
           "accuracy": float(accuracy_score(y, flagged)),
           "precision": float(precision_score(y, flagged, zero_division=0)),
           "recall": float(recall_score(y, flagged, zero_division=0))}
    two_classes = 0 < y.sum() < len(y)
    out["AUC"] = float(roc_auc_score(y, score)) if two_classes else None
    out["AP"] = float(average_precision_score(y, score)) if two_classes else None
    return out


def window(frame: pd.DataFrame, origin: str):
    """Train on dates <= origin; test on the following 12 months that already have a target."""
    o = pd.Timestamp(origin)
    lab = frame.dropna(subset=["target_next"])
    train = lab[lab["date"] <= o]
    test = nm.complete(lab[(lab["date"] > o) & (lab["date"] <= o + pd.DateOffset(months=12))], FEATURES)
    return train, test


# ----------------------------------------------------------------------------- regional nowcast
def evaluate_regional(frame: pd.DataFrame) -> dict:
    folds, oof_eval, oof_last, primary = [], [], [], {}
    for origin in ORIGINS:
        train, test = window(frame, origin)
        actual = test["target_next"].to_numpy()
        preds = {
            "Persistence (this month carried forward)": test["infl"].to_numpy(),
            "3-month average": test["ma3"].to_numpy(),
            "Training-period mean": np.full(len(test), train["target_next"].mean()),
        }
        reg_base = nm.fit_regressor(train, BASE_FEATURES, "level")
        preds["Boosted trees, level target, no current rate (v1)"] = nm.predict_rate(
            reg_base, test, BASE_FEATURES, "level")
        reg_dep = nm.fit_regressor(train, FEATURES, "delta")
        dep_pred = nm.predict_rate(reg_dep, test, FEATURES, "delta")
        preds["Boosted trees, change target + current rate (deployed)"] = dep_pred

        clf = nm.fit_classifier(train, FEATURES)
        proba = clf.predict_proba(test[FEATURES])[:, 1]
        y = (test["target_next"] > TARGET_RATE).astype(int).to_numpy()
        fold = {
            "origin": origin,
            "window": f"{(pd.Timestamp(origin) + pd.DateOffset(months=1)):%Y-%m}..{test['date'].max():%Y-%m}",
            "n": int(len(test)), "mean_actual": float(actual.mean()),
            "regression": {k: reg_metrics(actual, v) for k, v in preds.items()},
            "classifier": {
                "Persistence rule (this month > 4%)": cls_metrics(y, test["infl"].to_numpy(),
                                                                  (test["infl"] > TARGET_RATE).astype(int)),
                "Boosted trees classifier (deployed)": cls_metrics(y, proba, (proba >= 0.5).astype(int)),
            },
        }
        folds.append(fold)
        x = test["infl"].to_numpy()
        fold["shrinkage_mae"] = {str(l): mae(actual, x + l * (dep_pred - x)) for l in LAMBDAS}
        err = actual - x                      # error of the carry-forward estimate
        if origin != PRIMARY:
            oof_eval.append(err)
        else:
            oof_last.append(err)
            primary = {"train": train, "test": test, "reg": reg_dep, "pred": dep_pred,
                       "proba": proba, "y": y, "actual": actual}
    primary["q"] = quantiles(oof_eval)                    # honest: calibrated before the primary window
    primary["q_prod"] = quantiles(oof_eval + oof_last)    # production: every past window
    return {"folds": folds, "primary": primary}


def write_primary_artifacts(res: dict) -> dict:
    p = res["primary"]
    test, pred, q = p["test"], p["pred"], p["q"]
    df = pd.DataFrame({"date": test["date"].to_numpy(), "geolocation": test["geolocation"].to_numpy(),
                       "actual": p["actual"], "pred": test["infl"].to_numpy(), "model_pred": pred,
                       "infl_now": test["infl"].to_numpy()})
    df["residual"] = df["actual"] - df["pred"]   # headline estimate = this month carried forward
    df["lo80"], df["hi80"] = df["pred"] + q["q10"], df["pred"] + q["q90"]
    df["lo95"], df["hi95"] = df["pred"] + q["q025"], df["pred"] + q["q975"]
    df.to_parquet(OUT / "nowcast_test.parquet", index=False)
    cov80 = float(((df["actual"] >= df["lo80"]) & (df["actual"] <= df["hi80"])).mean())
    cov95 = float(((df["actual"] >= df["lo95"]) & (df["actual"] <= df["hi95"])).mean())

    pd.DataFrame({"date": test["date"].to_numpy(), "geolocation": test["geolocation"].to_numpy(),
                  "y_true": p["y"], "proba": p["proba"]}).to_parquet(OUT / "clf_test.parquet", index=False)
    fpr, tpr, _ = roc_curve(p["y"], p["proba"])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_parquet(OUT / "roc_curve.parquet", index=False)
    prec, rec, _ = precision_recall_curve(p["y"], p["proba"])
    pd.DataFrame({"precision": prec, "recall": rec}).to_parquet(OUT / "pr_curve.parquet", index=False)

    import shap  # only needed offline; the API never imports it
    explainer = shap.TreeExplainer(p["reg"])
    sv = explainer.shap_values(test[FEATURES])
    long = pd.DataFrame({
        "date": np.repeat(test["date"].to_numpy(), len(FEATURES)),
        "geolocation": np.repeat(test["geolocation"].to_numpy(), len(FEATURES)),
        "feature": np.tile(FEATURES, len(test)),
        "feature_value": test[FEATURES].to_numpy().ravel(),
        "shap_value": sv.ravel()})
    long.to_parquet(OUT / "shap_values.parquet", index=False)
    imp = (long.assign(a=long["shap_value"].abs()).groupby("feature")["a"].mean()
           .sort_values(ascending=False).rename("mean_abs_shap").reset_index())
    imp.to_parquet(OUT / "feature_importance.parquet", index=False)
    return {"coverage80": cov80, "coverage95": cov95, "shap_base": float(np.ravel(explainer.expected_value)[0])}


def write_latest(frame: pd.DataFrame, q: dict) -> int:
    """Latest month: headline estimate is carried forward; the boosted model supplies the
    breach probability and an explainable adjustment."""
    labelled = frame.dropna(subset=["target_next"])
    reg = nm.fit_regressor(labelled, FEATURES, "delta")
    clf = nm.fit_classifier(labelled, FEATURES)
    latest = frame[frame["date"] == frame["date"].max()].copy()
    latest["model_pred"] = nm.predict_rate(reg, latest, FEATURES, "delta")
    latest["proba_above4"] = clf.predict_proba(latest[FEATURES])[:, 1]
    now = latest["infl"].to_numpy()
    out = pd.DataFrame({
        "geolocation": latest["geolocation"].to_numpy(), "date": latest["date"].to_numpy(),
        "infl_now": now, "pred_next": now, "model_pred": latest["model_pred"].to_numpy(),
        "model_adjustment": latest["model_pred"].to_numpy() - now,
        "lo80": now + q["q10"], "hi80": now + q["q90"], "proba_above4": latest["proba_above4"].to_numpy()})
    out.to_parquet(OUT / "nowcast_latest.parquet", index=False)
    return len(out)


# ----------------------------------------------------------------------------- national rolling backtest
def national_rolling() -> dict:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cpi = fsvc.national_cpi().copy()
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi = cpi.reset_index(drop=True)
    cpi["yoy"] = cpi["cpi"].pct_change(12) * 100
    folds = []
    for origin in ORIGINS:
        o = pd.Timestamp(origin)
        hist = cpi[cpi["date"] <= o]
        fut = cpi[(cpi["date"] > o) & (cpi["date"] <= o + pd.DateOffset(months=12))]
        h, n = hist["cpi"].to_numpy(), len(fut)
        actual = fut["yoy"].to_numpy()
        fold = {"origin": origin, "window": f"{fut['date'].min():%Y-%m}..{fut['date'].max():%Y-%m}", "models": {}}

        def pack(mean, l80, h80, l95, h95):
            y = fsvc._yoy_path(h, mean)
            out = reg_metrics(actual, y)
            if l80 is not None:
                a80, b80 = fsvc._yoy_path(h, l80), fsvc._yoy_path(h, h80)
                a95, b95 = fsvc._yoy_path(h, l95), fsvc._yoy_path(h, h95)
                out["coverage80"] = float(np.mean((actual >= a80) & (actual <= b80)))
                out["coverage95"] = float(np.mean((actual >= a95) & (actual <= b95)))
                out["width95"] = float(np.mean(b95 - a95))
            return out

        fit = SARIMAX(np.log(h), order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.get_forecast(steps=n)
        c80, c95 = np.exp(np.asarray(fc.conf_int(alpha=0.2))), np.exp(np.asarray(fc.conf_int(alpha=0.05)))
        fold["models"]["SARIMA"] = pack(np.exp(np.asarray(fc.predicted_mean)),
                                        c80[:, 0], c80[:, 1], c95[:, 0], c95[:, 1])
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(hist["date"]), freq="MS")
            ets = ETSModel(pd.Series(h, index=idx), error="add", trend="add", damped_trend=True,
                           seasonal="add", seasonal_periods=12, initialization_method="estimated").fit(disp=False)
            pr = ets.get_prediction(start=fut["date"].iloc[0], end=fut["date"].iloc[-1])
            s80, s95 = pr.summary_frame(alpha=0.2), pr.summary_frame(alpha=0.05)
            fold["models"]["Holt-Winters"] = pack(s80["mean"].to_numpy(), s80["pi_lower"].to_numpy(),
                                                  s80["pi_upper"].to_numpy(), s95["pi_lower"].to_numpy(),
                                                  s95["pi_upper"].to_numpy())
        except Exception as exc:  # keep the evaluation honest but do not abort the build
            fold["models"]["Holt-Winters"] = {"error": str(exc)}
        last_yoy = float(hist["yoy"].iloc[-1])
        fold["models"]["Naive (last year-on-year rate held)"] = reg_metrics(actual, np.full(n, last_yoy))
        folds.append(fold)

    names = list(folds[0]["models"])
    mean = {}
    for m in names:
        rows = [f["models"][m] for f in folds if "MAE" in f["models"].get(m, {})]
        keys = {k for r in rows for k in r}
        mean[m] = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    return {"folds": folds, "mean": mean}


# ----------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = nm.load_features()
    print("regional rolling-origin evaluation ...")
    res = evaluate_regional(frame)
    extra = write_primary_artifacts(res)
    n_latest = write_latest(frame, res["primary"]["q_prod"])
    print("national rolling backtest ...")
    nat = national_rolling()

    primary = res["folds"][-1]
    dep_key = "Boosted trees, change target + current rate (deployed)"
    p = res["primary"]
    ss_res = float(np.sum((p["actual"] - p["pred"]) ** 2))
    ss_tot = float(np.sum((p["actual"] - p["actual"].mean()) ** 2))
    cm = primary["classifier"]["Boosted trees classifier (deployed)"]

    path = OUT / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics.update({
        "trained_at": datetime.now().replace(microsecond=0).isoformat(),
        "gbm": {**primary["regression"][dep_key], "R2": 1 - ss_res / ss_tot},
        "headline_nowcast": {**primary["regression"]["Persistence (this month carried forward)"],
                             "why": "carry-forward beat every boosted-trees variant in all four windows"},
        "classifier": {"AUC": cm["AUC"], "average_precision": cm["AP"], "accuracy": cm["accuracy"],
                       "precision": cm["precision"], "recall": cm["recall"],
                       "positive_rate": cm["positive_rate"], "n_test": cm["n"]},
        "classifier_baseline": {**{k: primary["classifier"]["Persistence rule (this month > 4%)"][k]
                                   for k in ("AUC", "average_precision" if False else "AP", "accuracy", "precision", "recall")},
                                "name": "Persistence rule (this month > 4%)"},
        "shap_base_value": extra["shap_base"], "features": FEATURES, "target_rate": TARGET_RATE,
        "nowcast": {
            "spec": "GradientBoostingRegressor on the month-to-month change, features incl. current rate",
            "primary_window": primary["window"], "folds": res["folds"],
            "interval_production": p["q_prod"],
            "interval": {**p["q"], "coverage80_primary": extra["coverage80"],
                         "coverage95_primary": extra["coverage95"],
                         "calibrated_on": "windows ending " + ", ".join(o[:7] for o in ORIGINS[:-1])},
        },
        "national_rolling": nat,
    })
    path.write_text(json.dumps(clean(metrics), indent=1))

    print(f"\nPrimary window {primary['window']} (n={primary['n']}) -- regional nowcast MAE / RMSE (pp)")
    for k, v in primary["regression"].items():
        print(f"  {k:62s} {v['MAE']:.3f} / {v['RMSE']:.3f}")
    print("Breach classifier:")
    for k, v in primary["classifier"].items():
        print(f"  {k:40s} AUC {v['AUC']:.3f}  AP {v['AP']:.3f}  acc {v['accuracy']:.3f}  prec {v['precision']:.2f}  rec {v['recall']:.2f}")
    print(f"Nowcast 80% / 95% interval coverage on primary window: {extra['coverage80']:.2f} / {extra['coverage95']:.2f}")
    print("National mean over 4 folds:", {k: {a: round(b, 2) for a, b in v.items() if a in ('MAE', 'coverage80', 'coverage95')} for k, v in nat['mean'].items()})
    print(f"latest-month rows scored: {n_latest}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
