import numpy as np, pandas as pd, pytest
from src import forecast as F, nowcast as N
from src.db import connect


@pytest.fixture(scope="module")
def con():
    return connect()


def test_warehouse_shape(con):
    assert con.execute("SELECT count(*) FROM fact_cpi").fetchone()[0] >= 1000
    assert con.execute("SELECT count(*) FROM dim_region").fetchone()[0] == 19


def test_quality_checks_pass(con):
    from src.db import ROOT
    rows = con.execute((ROOT / "sql" / "02_quality_checks.sql").read_text()).fetchall()
    assert all(v == 0 for _, v in rows), rows


def test_features_no_leakage(con):
    f = con.execute("SELECT * FROM features WHERE NOT is_national ORDER BY region_id, month_start").df()
    nxt = f.groupby("region_id")["yoy_all"].shift(-1)
    ok = f["target_yoy"].notna()
    assert np.allclose(f.loc[ok, "target_yoy"], nxt[ok])          # the target is strictly next month's value
    assert f.groupby("region_id")["target_yoy"].apply(lambda s: s.iloc[-1:].isna().all()).all()


def test_forecast_bands_nest(con):
    fc = F.forecast(F.national_cpi(con), 6)
    assert (fc["yoy_lo95"] <= fc["yoy_lo80"]).all() and (fc["yoy_hi80"] <= fc["yoy_hi95"]).all()


def test_shock_moves_forecast_up(con):
    cpi = F.national_cpi(con); fc = F.forecast(cpi, 6)
    sh = F.apply_shock(fc, cpi, 38.0, 10.0)
    assert (sh["yoy"] > fc["yoy"]).all()
    assert abs((sh["yoy"] - fc["yoy"]).iloc[0] - 3.8 * (1 + fc["yoy"].iloc[0] / 100)) < 0.5


def test_models_train(con):
    p = N.usable(N.load_panel(con)); tr = p[(p.month_start < N.TEST_START) & p.target_yoy.notna()]
    reg, clf = N.fit_models(tr, n_estimators=30)
    out = N.predict(reg, clf, p.head(20))
    assert out["p_breach"].between(0, 1).all()
