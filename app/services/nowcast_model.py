"""Regional next-month nowcast: features, model definitions and live what-if scoring.

Shared by the API (lazy, in-process scoring) and ``models/train_models.py`` (offline
evaluation + artifact generation), so the two can never drift apart.

Design (see README, "Models"):
  * Target: next month's year-on-year inflation for one of the 18 regions.
  * Deployed regressor predicts the *change* from this month (a learned correction on top of
    the persistence baseline), so it can extrapolate in a regime the training data never saw.
  * Classifier predicts P(next month > 4%), the upper edge of the BSP target band.
  * The latest observed rate (``infl``) is a feature; the first version of the model left it out.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from app.config import settings

TARGET_RATE = 4.0
SEED = 7
BASE_FEATURES = ["lag1", "lag2", "lag3", "lag6", "lag12", "ma3", "ma6", "ma12",
                 "vol12", "month_sin", "month_cos", "time_idx"]
FEATURES = ["infl"] + BASE_FEATURES


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(settings.warehouse / "feat_region_month.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["geolocation", "date"]).reset_index(drop=True)


def complete(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rows whose features are all observed (drops the first year of each region's history)."""
    return df.dropna(subset=features)


def make_regressor() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3,
                                     subsample=0.8, min_samples_leaf=10, random_state=SEED)


def make_classifier() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                      subsample=0.8, min_samples_leaf=10, random_state=SEED)


def fit_regressor(train: pd.DataFrame, features: list[str], mode: str = "delta"):
    train = complete(train, features)
    y = train["target_next"] - train["infl"] if mode == "delta" else train["target_next"]
    return make_regressor().fit(train[features], y)


def predict_rate(model, rows: pd.DataFrame, features: list[str], mode: str = "delta") -> np.ndarray:
    raw = model.predict(rows[features])
    return rows["infl"].to_numpy() + raw if mode == "delta" else raw


def fit_classifier(train: pd.DataFrame, features: list[str]):
    train = complete(train, features)
    return make_classifier().fit(train[features], (train["target_next"] > TARGET_RATE).astype(int))


class Nowcaster:
    """Production models, trained on every labelled month (through the last published target)."""

    def __init__(self) -> None:
        self.frame = load_features()
        labelled = self.frame.dropna(subset=["target_next"])
        self.reg = fit_regressor(labelled, FEATURES, "delta")
        self.clf = fit_classifier(labelled, FEATURES)
        self.last_date = self.frame["date"].max()

    def latest_row(self, region: str) -> pd.Series | None:
        rows = self.frame[(self.frame["geolocation"] == region) & (self.frame["date"] == self.last_date)]
        return rows.iloc[0] if len(rows) else None

    def score(self, row: dict | pd.Series) -> tuple[float, float]:
        x = pd.DataFrame([row])[FEATURES]
        pred = float(x["infl"].iloc[0] + self.reg.predict(x)[0])
        proba = float(self.clf.predict_proba(x)[0, 1])
        return pred, proba

    def what_if(self, region: str, infl_now: float) -> dict | None:
        """Re-score the region's latest month as if this month's rate had been ``infl_now``.

        Lags are history and stay fixed; the rolling statistics are recomputed exactly from
        the region's own series with the last value replaced.
        """
        series = self.frame[(self.frame["geolocation"] == region)].sort_values("date")
        if series.empty:
            return None
        base = series.iloc[-1].copy()
        scen = base.copy()
        vals = series["infl"].to_numpy().copy()
        vals[-1] = infl_now
        scen["infl"] = infl_now
        scen["ma3"] = float(np.mean(vals[-3:]))
        scen["ma6"] = float(np.mean(vals[-6:]))
        scen["ma12"] = float(np.mean(vals[-12:]))
        scen["vol12"] = float(np.std(vals[-12:], ddof=1))
        p0, pr0 = self.score(base)
        p1, pr1 = self.score(scen)
        return {"region": region, "date": base["date"].strftime("%Y-%m-%d"),
                "actual_now": float(base["infl"]), "scenario_now": float(infl_now),
                "baseline_next": p0, "baseline_proba": pr0,
                "scenario_next": p1, "scenario_proba": pr1,
                "features": {k: float(scen[k]) for k in FEATURES}}


@lru_cache(maxsize=1)
def nowcaster() -> Nowcaster:
    return Nowcaster()
