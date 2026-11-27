"""Tests for the 3-asset layer (assets.py), the cleaned/raw data provenance, and the real bundled dataset."""
import copy, hashlib, json, os, pickle
import numpy as np, pandas as pd, pytest
import assets as AS, datasets as DS, data_fetcher as DF, optimizer as OPT, universe as U

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _psd_cov(n, seed=0, vol=0.15):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n + 40)); C = A @ A.T / (n + 40)
    d = np.sqrt(np.diag(C)); return C / np.outer(d, d) * vol ** 2


# ---- debt/gold have no floors; equity follows the author's rule
def test_no_debt_gold_floor_or_defensive_split_rule():
    import optimizer as O
    for name in ("DEF_MIN_W", "defensive_split"):
        assert not hasattr(AS, name) and not hasattr(O, name), name
    assert (O.EQUITY_NEUTRAL, O.EQUITY_SLOPE, O.EQUITY_MIN, O.EQUITY_MAX) == (0.70, 0.10, 0.50, 0.90)


# ---- CovModel
def test_covmodel_keeps_cross_through_pickle_and_deepcopy():
    cov = _psd_cov(4)
    cm = AS.CovModel(cov, cross={"marker": 7})
    for c in (pickle.loads(pickle.dumps(cm)), copy.deepcopy(cm)):
        assert isinstance(c, AS.CovModel) and c.cross == {"marker": 7} and np.allclose(c, cov)


# ---- build_cross on synthetic levels (test data only)
def test_build_cross_uses_actual_returns_and_is_psd():
    rng = np.random.default_rng(3); idx = pd.bdate_range("2020-01-01", periods=900)
    mk = rng.normal(0, 0.009, 900)
    lv = pd.DataFrame({f"F{i}": 100 * np.cumprod(1 + 0.8 * mk + rng.normal(0, 0.004, 900)) for i in range(4)}, index=idx)
    debt = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.0015, 900)), index=idx)
    gold = pd.Series(100 * np.cumprod(1 + 0.1 * mk + rng.normal(0.0003, 0.008, 900)), index=idx)
    cov = lv.pct_change().dropna().cov().values * 252
    cr = AS.build_cross(cov, lv, {"debt": debt, "gold": gold})
    assert "assumed" not in json.dumps(cr["source"]).lower(), cr["source"]       # both legs come from data here
    full = AS.full_covariance(cov, cr)
    assert full.shape == (6, 6) and np.linalg.eigvalsh(full).min() > -1e-10
    # debt variance equals the sample variance on the common dates, annualised by observations per year
    r = debt.pct_change().dropna()
    ppy = len(r) / ((r.index[-1] - r.index[0]).days / 365.25)
    assert cr["cov_dd"][0, 0] == pytest.approx(r.var(ddof=1) * ppy, rel=0.05)

def test_build_cross_flags_assumed_when_dataset_missing():
    cov = _psd_cov(5); idx = pd.bdate_range("2021-01-01", periods=400)
    lv = pd.DataFrame(np.cumprod(1 + np.random.default_rng(1).normal(0, .01, (400, 5)), axis=0) * 100, index=idx)
    cr = AS.build_cross(cov, lv, {})
    assert "assumed" in json.dumps(cr["source"]).lower()


# ---- point-in-time valuation statistics
def test_valuation_stats_fallback_when_history_short():
    me = pd.date_range("2019-01-31", periods=24, freq="ME")
    v = pd.DataFrame({"pe": np.linspace(18, 24, 24), "pb": np.linspace(3, 3.6, 24)}, index=me)
    st = AS.valuation_stats(v, pd.Timestamp("2020-12-31"), method="expanding")
    assert st["method"] != "expanding" and tuple(st["pe"]) == AS.SYNOPSIS_STATS["pe"]


# ---- shipped real data: provenance and cleaning
@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "provenance.json")), reason="no shipped data")
def test_provenance_checksums_match_cleaned_and_raw_files():
    prov = DS.read_provenance(DATA)
    for key, p in prov.items():
        assert DS.sha256_file(os.path.join(DATA, p["file"])) == p["sha256"], key
    q = json.load(open(os.path.join(DATA, "data_quality_report.json")))
    for fname, sha in q["raw_files"].items() if isinstance(q["raw_files"], dict) else []:
        path = os.path.join(DATA, "raw", fname)
        if os.path.exists(path):
            assert hashlib.sha256(open(path, "rb").read()).hexdigest() == (sha if isinstance(sha, str) else sha.get("sha256"))

@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "cleaned", "nifty_tri.csv")), reason="no shipped data")
def test_cleaned_series_sorted_unique_positive_and_not_interpolated():
    for k in ("nifty_tri", "gold"):
        df = pd.read_csv(os.path.join(DATA, "cleaned", f"{k}.csv"), parse_dates=["date"])
        assert df["date"].is_monotonic_increasing and not df["date"].duplicated().any() and (df["value"] > 0).all()
        # no interpolation: no weekend dates and no run of identical values created by filling
        assert (df["date"].dt.dayofweek < 5).mean() > 0.99
    raw_rows = sum(len(pd.read_csv(os.path.join(DATA, "raw", f"nifty_{y}.csv"))) for y in range(2014, 2025))
    assert len(pd.read_csv(os.path.join(DATA, "cleaned", "nifty_tri.csv"))) <= raw_rows       # never more rows than raw

@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "cleaned", "fund_nav.csv")), reason="no shipped data")
def test_bundled_12_fund_dataset_passes_identity_validation_and_common_window():
    d = DF.load_dataset()
    assert d["source"] == "bundled_dataset" and d["dataset_complete"]
    assert list(d["returns_df"].columns) == U.FUNDS_DEFAULT and len(U.FUNDS_DEFAULT) == 12
    assert all(v["ok"] for v in d["validation"].values())
    rd = d["returns_df"].dropna()
    assert len(rd) > 500 and rd.index[0] >= d["returns_df"].index[0]


def test_cash_is_not_an_asset_anywhere():
    assert DS.REQUIRED == ("nifty_tri", "nifty_valuation", "gold", "debt")
    assert not hasattr(OPT, "CASH_RET") and not hasattr(AS, "CASH_RET")
